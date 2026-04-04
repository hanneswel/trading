import logging
from datetime import datetime, timezone

import httpx

from .http import RateLimiter, gamma_limiter, request_with_retry
from .models import Market

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


def _parse_market(raw: dict) -> Market | None:
    try:
        clob_token_ids = raw.get("clobTokenIds") or []
        if isinstance(clob_token_ids, str):
            import json
            clob_token_ids = json.loads(clob_token_ids)

        outcome_prices_raw = raw.get("outcomePrices") or raw.get("outcome_prices") or "[]"
        if isinstance(outcome_prices_raw, str):
            import json
            outcome_prices_raw = json.loads(outcome_prices_raw)
        outcome_prices = [float(p) for p in outcome_prices_raw]

        outcomes_raw = raw.get("outcomes") or []
        if isinstance(outcomes_raw, str):
            import json
            outcomes_raw = json.loads(outcomes_raw)

        return Market(
            id=str(raw["id"]),
            question=raw.get("question", ""),
            condition_id=raw.get("conditionId") or raw.get("condition_id", ""),
            slug=raw.get("slug", ""),
            description=raw.get("description", ""),
            outcomes=outcomes_raw,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            volume=float(raw.get("volumeNum") or raw.get("volume") or 0),
            volume_24hr=float(raw.get("volume24hr") or 0),
            liquidity=float(raw.get("liquidityNum") or raw.get("liquidity") or 0),
            active=bool(raw.get("active", False)),
            closed=bool(raw.get("closed", False)),
            end_date=raw.get("endDate") or raw.get("end_date"),
            spread=_safe_float(raw.get("spread")),
            best_bid=_safe_float(raw.get("bestBid")),
            best_ask=_safe_float(raw.get("bestAsk")),
            last_trade_price=_safe_float(raw.get("lastTradePrice")),
            fetched_at=datetime.now(timezone.utc),
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to parse market %s: %s", raw.get("id", "?"), e)
        return None


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


async def fetch_markets(
    client: httpx.AsyncClient,
    limiter: RateLimiter = gamma_limiter,
    *,
    volume_min: float = 10_000,
    page_size: int = 100,
) -> list[Market]:
    """Fetch all active markets from Gamma API with pagination."""
    markets: list[Market] = []
    offset = 0

    while True:
        params = {
            "limit": page_size,
            "offset": offset,
            "active": "true",
            "closed": "false",
            "order": "volume",
            "ascending": "false",
        }
        if volume_min > 0:
            params["volume_num_min"] = str(volume_min)

        data = await request_with_retry(
            client, "GET", f"{GAMMA_BASE}/markets", limiter, params=params
        )

        if not data:
            break

        for raw in data:
            market = _parse_market(raw)
            if market and market.clob_token_ids:
                markets.append(market)

        logger.info("Fetched page at offset %d: %d markets", offset, len(data))

        if len(data) < page_size:
            break
        offset += page_size

    logger.info("Total markets fetched: %d", len(markets))
    return markets
