import asyncio
import logging
import sys

import click
import httpx

from .clob import fetch_all_order_books
from .db import DEFAULT_DB, get_top_markets, init_db, insert_order_books, upsert_markets
from .estimator import estimate_markets
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
@click.option("--ttl", default=6.0, type=float, help="Cache TTL in hours.")
@click.option("--threshold", default=0.10, type=float, help="Signal divergence threshold.")
@click.option("--dry-run", is_flag=True, help="Show what would be estimated without making API calls.")
@click.pass_context
def estimate(ctx: click.Context, top: int, sort: str, model: str, ttl: float, threshold: float, dry_run: bool) -> None:
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
            markets, conn, model=model, ttl_hours=ttl
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
@click.option("--model", default="claude-sonnet-4-20250514", help="Anthropic model to use.")
@click.option("--ttl", default=6.0, type=float, help="Cache TTL in hours.")
@click.option("--threshold", default=0.10, type=float, help="Signal divergence threshold.")
@click.option("--max-markets", default=50, type=int, help="Maximum markets to scan.")
@click.option("--dry-run", is_flag=True, help="Show what would be scanned without making API calls.")
@click.pass_context
def scan(ctx: click.Context, volume_min: float, model: str, ttl: float, threshold: float, max_markets: int, dry_run: bool) -> None:
    """Scan all liquid markets for mispricing signals."""
    db_path = ctx.obj["db"]
    conn = init_db(db_path)

    # Get all active markets above volume threshold
    rows = conn.execute(
        "SELECT * FROM markets WHERE active = 1 AND closed = 0 AND volume >= ? ORDER BY volume DESC LIMIT ?",
        (volume_min, max_markets),
    ).fetchall()

    if not rows:
        click.echo(f"No active markets with volume >= ${volume_min:,.0f}. Run 'polymarket fetch' first.")
        conn.close()
        return

    from .db import _row_to_market
    markets = [_row_to_market(r) for r in rows]

    # Filter out markets where outcome is near-certain (price < 0.03 or > 0.97)
    # These are not interesting for estimation
    markets = [
        m for m in markets
        if m.outcome_prices and 0.03 <= m.outcome_prices[0] <= 0.97
    ]

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

    click.echo(f"Scanning {len(markets)} markets for mispricing signals...\n")

    async def _run():
        estimations = await estimate_markets(
            markets, conn, model=model, ttl_hours=ttl
        )

        signals = detect_signals(estimations, threshold)

        if signals:
            click.echo(f"\n{'='*92}")
            click.echo(f" SIGNALS FOUND: {len(signals)} (out of {len(estimations)} estimated)")
            click.echo(f"{'='*92}\n")
            click.echo(f"{'#':>2}  {'Question':<40} {'Side':<8} {'Mkt':>5} {'Model':>6} {'Div':>7} {'EV':>7} {'Conf':>6}")
            click.echo("-" * 87)
            for i, sig in enumerate(signals, 1):
                q = sig.market_question[:37] + "..." if len(sig.market_question) > 40 else sig.market_question
                click.echo(
                    f"{i:>2}  {q:<40} {sig.side:<8} {sig.market_price:>5.2f} {sig.model_probability:>6.2f} "
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

        # Cost summary
        total_input = sum(e.input_tokens for e in estimations)
        total_output = sum(e.output_tokens for e in estimations)
        total_searches = sum(e.search_count for e in estimations)
        cost = (total_input / 1_000_000 * 3) + (total_output / 1_000_000 * 15) + (total_searches / 1000 * 10)
        click.echo(f"\nScan complete: {len(estimations)} markets | {len(signals)} signals | Est. cost: ${cost:.3f}")

    asyncio.run(_run())
    conn.close()
