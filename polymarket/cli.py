import asyncio
import logging
import sys

import click
import httpx

from .clob import fetch_all_order_books
from .db import DEFAULT_DB, get_top_markets, init_db, insert_order_books, upsert_markets
from .gamma import fetch_markets
from .http import clob_limiter, gamma_limiter


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
