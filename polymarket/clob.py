import asyncio
import logging
from datetime import datetime, timezone

import httpx

from .http import RateLimiter, clob_limiter, request_with_retry
from .models import Market, OrderBook, OrderLevel

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"


def _parse_levels(raw_levels: list[dict]) -> list[OrderLevel]:
    return [OrderLevel(price=float(l["price"]), size=float(l["size"])) for l in raw_levels]


async def fetch_order_book(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    token_id: str,
    outcome: str,
    condition_id: str,
) -> OrderBook | None:
    try:
        data = await request_with_retry(
            client, "GET", f"{CLOB_BASE}/book", limiter,
            params={"token_id": token_id},
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("Order book not found for token %s", token_id)
            return None
        raise

    bids = _parse_levels(data.get("bids") or [])
    asks = _parse_levels(data.get("asks") or [])

    # Calculate midpoint and spread from best bid/ask
    midpoint = None
    spread = None
    if bids and asks:
        best_bid = bids[0].price
        best_ask = asks[0].price
        midpoint = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

    return OrderBook(
        market_condition_id=condition_id,
        asset_id=token_id,
        outcome=outcome,
        bids=bids,
        asks=asks,
        midpoint=midpoint,
        spread=spread,
        last_trade_price=_safe_float(data.get("last_trade_price")),
        timestamp=data.get("timestamp"),
        fetched_at=datetime.now(timezone.utc),
    )


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


async def fetch_order_books_for_market(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    market: Market,
) -> list[OrderBook]:
    """Fetch order books for all outcomes of a market concurrently."""
    tasks = []
    for i, token_id in enumerate(market.clob_token_ids):
        outcome = market.outcomes[i] if i < len(market.outcomes) else f"Outcome {i}"
        tasks.append(
            fetch_order_book(client, limiter, token_id, outcome, market.condition_id)
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    books = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Error fetching order book for market %s: %s", market.condition_id, r)
        elif r is not None:
            books.append(r)
    return books


async def fetch_all_order_books(
    client: httpx.AsyncClient,
    limiter: RateLimiter = clob_limiter,
    markets: list[Market] = [],
    concurrency: int = 10,
) -> list[OrderBook]:
    """Fetch order books for multiple markets with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    all_books: list[OrderBook] = []

    async def fetch_with_sem(market: Market) -> list[OrderBook]:
        async with sem:
            return await fetch_order_books_for_market(client, limiter, market)

    tasks = [fetch_with_sem(m) for m in markets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("Error fetching books for market %s: %s", markets[i].condition_id, r)
        else:
            all_books.extend(r)

    logger.info("Fetched %d order books for %d markets", len(all_books), len(markets))
    return all_books
