import asyncio
import json
import logging
import sys

import click
import httpx

from .clob import fetch_all_order_books
from .db import DEFAULT_DB, get_top_markets, init_db, insert_order_books, upsert_markets
from .estimator import estimate_markets, screen_and_estimate_markets
from .gamma import fetch_markets
from .http import clob_limiter, gamma_limiter
from .signals import detect_signals


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _format_volume(vol: float) -> str:
    if vol >= 1_000_000:
        return f"${vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"${vol / 1_000:.0f}K"
    return f"${vol:.0f}"


@click.group()
@click.option("--db", default=DEFAULT_DB, help="SQLite database path.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, db: str, verbose: bool) -> None:
    """Polymarket data layer CLI."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


@main.command()
@click.option("--volume-min", default=10_000, type=float, help="Minimum volume filter (USD).")
@click.pass_context
def fetch(ctx: click.Context, volume_min: float) -> None:
    """Fetch active markets from Polymarket and store in SQLite."""
    db_path = ctx.obj["db"]

    async def _run() -> None:
        conn = init_db(db_path)
        async with httpx.AsyncClient(timeout=30) as client:
            markets = await fetch_markets(client, gamma_limiter, volume_min=volume_min)
        upsert_markets(conn, markets)
        conn.close()
        click.echo(f"Fetched and stored {len(markets)} markets (volume >= ${volume_min:,.0f})")

    asyncio.run(_run())


@main.command()
@click.option("--top", default=20, type=int, help="Number of top markets to fetch books for.")
@click.pass_context
def books(ctx: click.Context, top: int) -> None:
    """Fetch order books for top markets by volume."""
    db_path = ctx.obj["db"]

    async def _run() -> None:
        conn = init_db(db_path)
        markets = get_top_markets(conn, limit=top)
        if not markets:
            click.echo("No markets in database. Run 'polymarket fetch' first.")
            return

        async with httpx.AsyncClient(timeout=30) as client:
            order_books = await fetch_all_order_books(
                client, clob_limiter, markets=markets
            )
        insert_order_books(conn, order_books)
        conn.close()
        click.echo(f"Fetched {len(order_books)} order books for {len(markets)} markets")

    asyncio.run(_run())


@main.command()
@click.option("--top", default=20, type=int, help="Number of markets to display.")
@click.option("--sort", default="volume", type=click.Choice(["volume", "liquidity", "volume_24hr"]))
@click.pass_context
def show(ctx: click.Context, top: int, sort: str) -> None:
    """Display top markets with implied probabilities."""
    db_path = ctx.obj["db"]
    conn = init_db(db_path)
    markets = get_top_markets(conn, limit=top, order_by=sort)
    conn.close()

    if not markets:
        click.echo("No markets in database. Run 'polymarket fetch' first.")
        return

    # Header
    click.echo(f"{'#':>3}  {'Question':<55} {'Yes':>6} {'No':>6} {'Volume':>10} {'Spread':>8}")
    click.echo("-" * 95)

    for i, m in enumerate(markets, 1):
        question = m.question[:52] + "..." if len(m.question) > 55 else m.question
        yes_price = m.outcome_prices[0] if len(m.outcome_prices) > 0 else 0
        no_price = m.outcome_prices[1] if len(m.outcome_prices) > 1 else 0
        spread = f"{m.spread:.3f}" if m.spread is not None else "  -"

        click.echo(
            f"{i:>3}  {question:<55} {yes_price:>6.2f} {no_price:>6.2f} {_format_volume(m.volume):>10} {spread:>8}"
        )

    click.echo(f"\nShowing {len(markets)} markets sorted by {sort}")
    if markets:
        click.echo(f"Data fetched at: {markets[0].fetched_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")


def _format_divergence(div: float) -> str:
    sign = "+" if div >= 0 else ""
    return f"{sign}{div:.2f}"


@main.command()
@click.option("--top", default=5, type=int, help="Number of top markets to estimate.")
@click.option("--sort", default="volume", type=click.Choice(["volume", "liquidity"]))
@click.option("--model", default="claude-sonnet-4-20250514", help="Anthropic model to use.")
@click.option("--ttl", default=12.0, type=float, help="Cache TTL in hours.")
@click.option("--threshold", default=0.10, type=float, help="Signal divergence threshold.")
@click.option("--max-searches", default=3, type=int, help="Max web searches per market.")
@click.option("--dry-run", is_flag=True, help="Show what would be estimated without making API calls.")
@click.pass_context
def estimate(ctx: click.Context, top: int, sort: str, model: str, ttl: float, threshold: float, max_searches: int, dry_run: bool) -> None:
    """Run LLM probability estimation on top markets."""
    db_path = ctx.obj["db"]
    conn = init_db(db_path)
    markets = get_top_markets(conn, limit=top, order_by=sort)

    if not markets:
        click.echo("No markets in database. Run 'polymarket fetch' first.")
        conn.close()
        return

    if dry_run:
        click.echo(f"Would estimate {len(markets)} markets using {model}:\n")
        for i, m in enumerate(markets, 1):
            question = m.question[:70] + "..." if len(m.question) > 70 else m.question
            yes_price = m.outcome_prices[0] if m.outcome_prices else 0
            click.echo(f"  {i:>2}. [{yes_price:.2f}] {question}")
        click.echo(f"\nModel: {model} | TTL: {ttl}h | Threshold: {threshold}")
        conn.close()
        return

    async def _run():
        estimations = await estimate_markets(
            markets, conn, model=model, ttl_hours=ttl, max_searches=max_searches
        )

        if not estimations:
            click.echo("No estimations completed.")
            return

        # Display results
        click.echo(f"\n{'#':>2}  {'Question':<50} {'Mkt':>5} {'Model':>6} {'Div':>7} {'Conf':>6}")
        click.echo("-" * 82)

        for i, est in enumerate(estimations, 1):
            q = est.market_question[:47] + "..." if len(est.market_question) > 50 else est.market_question
            div = est.model_probability - est.market_price
            click.echo(
                f"{i:>2}  {q:<50} {est.market_price:>5.2f} {est.model_probability:>6.2f} "
                f"{_format_divergence(div):>7} {est.confidence:>6}"
            )

        # Show signals
        signals = detect_signals(estimations, threshold)
        if signals:
            click.echo(f"\n--- SIGNALS (divergence > {threshold}) ---\n")
            click.echo(f"{'#':>2}  {'Question':<45} {'Side':<8} {'Mkt':>5} {'Model':>6} {'Div':>7} {'EV':>7} {'Conf':>6}")
            click.echo("-" * 92)
            for i, sig in enumerate(signals, 1):
                q = sig.market_question[:42] + "..." if len(sig.market_question) > 45 else sig.market_question
                click.echo(
                    f"{i:>2}  {q:<45} {sig.side:<8} {sig.market_price:>5.2f} {sig.model_probability:>6.2f} "
                    f"{_format_divergence(sig.divergence):>7} {sig.expected_value:>+7.3f} {sig.confidence:>6}"
                )
        else:
            click.echo(f"\nNo signals found (threshold: {threshold})")

        # Cost summary
        total_input = sum(e.input_tokens for e in estimations)
        total_output = sum(e.output_tokens for e in estimations)
        total_searches = sum(e.search_count for e in estimations)
        # Sonnet pricing: $3/1M input, $15/1M output, $10/1K searches
        cost = (total_input / 1_000_000 * 3) + (total_output / 1_000_000 * 15) + (total_searches / 1000 * 10)
        click.echo(f"\nTokens: {total_input:,} in / {total_output:,} out | Searches: {total_searches} | Est. cost: ${cost:.3f}")

    asyncio.run(_run())
    conn.close()


@main.command()
@click.option("--volume-min", default=50_000, type=float, help="Minimum volume for scan.")
@click.option("--model", default="claude-sonnet-4-20250514", help="Anthropic model for deep estimation.")
@click.option("--ttl", default=12.0, type=float, help="Cache TTL in hours.")
@click.option("--threshold", default=0.10, type=float, help="Signal divergence threshold.")
@click.option("--max-markets", default=50, type=int, help="Maximum markets to scan.")
@click.option("--max-searches", default=3, type=int, help="Max web searches per market (deep stage).")
@click.option("--pre-threshold", default=0.08, type=float, help="Screen divergence threshold to promote to deep.")
@click.option("--no-screen", is_flag=True, help="Skip screening stage, run deep estimation on all markets.")
@click.option("--dry-run", is_flag=True, help="Show what would be scanned without making API calls.")
@click.pass_context
def scan(ctx: click.Context, volume_min: float, model: str, ttl: float, threshold: float, max_markets: int, max_searches: int, pre_threshold: float, no_screen: bool, dry_run: bool) -> None:
    """Scan all liquid markets for mispricing signals. Uses two-stage pipeline by default."""
    db_path = ctx.obj["db"]
    conn = init_db(db_path)

    # Get all active markets above volume threshold, then filter by price
    from .db import _row_to_market
    rows = conn.execute(
        "SELECT * FROM markets WHERE active = 1 AND closed = 0 AND volume >= ? ORDER BY volume DESC",
        (volume_min,),
    ).fetchall()

    # Filter out near-certain outcomes (not interesting for estimation), then apply limit
    markets = [
        _row_to_market(r) for r in rows
        if json.loads(r["outcome_prices"])[0] is not None
        and 0.03 <= json.loads(r["outcome_prices"])[0] <= 0.97
    ][:max_markets]

    if not markets:
        click.echo(f"No interesting markets with volume >= ${volume_min:,.0f} (price 0.03-0.97). Run 'polymarket fetch' first.")
        conn.close()
        return

    if dry_run:
        click.echo(f"Would scan {len(markets)} markets (volume >= ${volume_min:,.0f}) using {model}:\n")
        for i, m in enumerate(markets[:20], 1):
            q = m.question[:65] + "..." if len(m.question) > 65 else m.question
            yes = m.outcome_prices[0] if m.outcome_prices else 0
            click.echo(f"  {i:>3}. [{yes:.2f}] {_format_volume(m.volume):>8} {q}")
        if len(markets) > 20:
            click.echo(f"  ... and {len(markets) - 20} more")
        click.echo(f"\nModel: {model} | TTL: {ttl}h | Threshold: {threshold}")
        conn.close()
        return

    if no_screen:
        click.echo(f"Scanning {len(markets)} markets (deep only, no screening)...\n")
    else:
        click.echo(f"Scanning {len(markets)} markets (screen → deep pipeline)...\n")

    async def _run():
        if no_screen:
            estimations = await estimate_markets(
                markets, conn, model=model, ttl_hours=ttl, max_searches=max_searches
            )
        else:
            estimations = await screen_and_estimate_markets(
                markets, conn, deep_model=model, ttl_hours=ttl,
                max_searches=max_searches, pre_threshold=pre_threshold,
            )

        signals = detect_signals(estimations, threshold)

        if signals:
            click.echo(f"\n{'='*92}")
            click.echo(f" SIGNALS FOUND: {len(signals)} (out of {len(estimations)} estimated)")
            click.echo(f"{'='*92}\n")
            click.echo(f"{'#':>2} {'St':>2}  {'Question':<38} {'Side':<8} {'Mkt':>5} {'Model':>6} {'Div':>7} {'EV':>7} {'Conf':>6}")
            click.echo("-" * 90)
            for i, sig in enumerate(signals, 1):
                q = sig.market_question[:35] + "..." if len(sig.market_question) > 38 else sig.market_question
                # Find the stage from the matching estimation
                stage_char = "D"
                for e in estimations:
                    if e.market_id == sig.market_id:
                        stage_char = "S" if e.stage == "screen" else "D"
                        break
                click.echo(
                    f"{i:>2}  {stage_char:>2} {q:<38} {sig.side:<8} {sig.market_price:>5.2f} {sig.model_probability:>6.2f} "
                    f"{_format_divergence(sig.divergence):>7} {sig.expected_value:>+7.3f} {sig.confidence:>6}"
                )
            # Show reasoning for top signals
            click.echo(f"\n--- Top Signal Details ---\n")
            for sig in signals[:5]:
                click.echo(f"  {sig.market_question}")
                click.echo(f"  {sig.side} | Market: {sig.market_price:.2f} | Model: {sig.model_probability:.2f} | EV: {sig.expected_value:+.3f}")
                click.echo(f"  {sig.reasoning}")
                click.echo()
        else:
            click.echo(f"\nNo signals found (threshold: {threshold}, markets: {len(estimations)})")

        # Cost summary with per-model pricing
        screen_ests = [e for e in estimations if e.stage == "screen"]
        deep_ests = [e for e in estimations if e.stage == "deep"]

        screen_input = sum(e.input_tokens for e in screen_ests)
        screen_output = sum(e.output_tokens for e in screen_ests)
        # Haiku pricing: $0.80/1M input, $4/1M output
        screen_cost = (screen_input / 1_000_000 * 0.80) + (screen_output / 1_000_000 * 4)

        deep_input = sum(e.input_tokens for e in deep_ests)
        deep_output = sum(e.output_tokens for e in deep_ests)
        deep_searches = sum(e.search_count for e in deep_ests)
        # Sonnet pricing: $3/1M input, $15/1M output, $10/1K searches
        deep_cost = (deep_input / 1_000_000 * 3) + (deep_output / 1_000_000 * 15) + (deep_searches / 1000 * 10)

        total_cost = screen_cost + deep_cost
        click.echo(f"\nScreen: {len(screen_ests)} markets, ${screen_cost:.3f} | Deep: {len(deep_ests)} markets, ${deep_cost:.3f} | Total: ${total_cost:.3f}")
        click.echo(f"Signals: {len(signals)} | Threshold: {threshold}")

    asyncio.run(_run())
    conn.close()
