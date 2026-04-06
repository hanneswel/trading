import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone

import anthropic

from .db import get_cached_estimation, insert_estimation
from .models import EstimationResult, Market
from .prompts import (
    ESTIMATION_JSON_SCHEMA,
    SCREEN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_screen_user_prompt,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
SCREEN_MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT = 5


def _parse_response(
    response, market: Market, model_name: str
) -> EstimationResult:
    """Parse Anthropic API response into EstimationResult."""
    # Extract text content and sources from response blocks
    text_parts = []
    sources = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
            if hasattr(block, "citations") and block.citations:
                for citation in block.citations:
                    if hasattr(citation, "url") and citation.url:
                        sources.append(citation.url)

    full_text = "\n".join(text_parts)

    # Parse JSON from response
    try:
        data = json.loads(full_text)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        start = full_text.find("{")
        end = full_text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(full_text[start:end])
        else:
            raise ValueError(f"Could not parse JSON from response: {full_text[:200]}")

    # Validate and clamp probability
    prob = float(data["probability"])
    if prob < 0 or prob > 1:
        logger.warning("Probability %.4f out of range for market %s, clamping", prob, market.id)
        prob = max(0.0, min(1.0, prob))

    # Validate confidence
    confidence = data.get("confidence", "medium").lower()
    if confidence not in ("low", "medium", "high"):
        confidence = "medium"

    # Extract usage info
    usage = response.usage
    input_tokens = usage.input_tokens if hasattr(usage, "input_tokens") else 0
    output_tokens = usage.output_tokens if hasattr(usage, "output_tokens") else 0
    search_count = 0
    if hasattr(usage, "server_tool_use") and usage.server_tool_use:
        search_count = getattr(usage.server_tool_use, "web_search_requests", 0)

    market_price = market.outcome_prices[0] if market.outcome_prices else 0.0

    return EstimationResult(
        market_id=market.id,
        market_question=market.question,
        model_probability=prob,
        confidence=confidence,
        reasoning=data.get("reasoning", ""),
        factors_for=data.get("factors_for", []),
        factors_against=data.get("factors_against", []),
        sources=list(dict.fromkeys(sources)),  # deduplicate, preserve order
        market_price=market_price,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        search_count=search_count,
        estimated_at=datetime.now(timezone.utc),
    )


def estimate_market(
    client: anthropic.Anthropic,
    market: Market,
    *,
    model: str = DEFAULT_MODEL,
    max_searches: int = 3,
) -> EstimationResult:
    """Run a single market estimation with web search."""
    user_prompt = build_user_prompt(market)

    logger.info("Estimating market: %s", market.question[:80])

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_searches,
            }
        ],
    )

    result = _parse_response(response, market, model)

    logger.info(
        "Market: %s | Model: %.2f | Market: %.2f | Div: %+.2f | Conf: %s | Searches: %d | Tokens: %d/%d",
        market.question[:50],
        result.model_probability,
        result.market_price,
        result.model_probability - result.market_price,
        result.confidence,
        result.search_count,
        result.input_tokens,
        result.output_tokens,
    )

    return result


def screen_market(
    client: anthropic.Anthropic,
    market: Market,
    *,
    model: str = SCREEN_MODEL,
) -> EstimationResult:
    """Run a cheap screening estimation without web search."""
    user_prompt = build_screen_user_prompt(market)

    logger.info("Screening market: %s", market.question[:80])

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SCREEN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    result = _parse_response(response, market, model)
    result.stage = "screen"

    logger.info(
        "Screen: %s | Model: %.2f | Market: %.2f | Div: %+.2f | Conf: %s",
        market.question[:50],
        result.model_probability,
        result.market_price,
        result.model_probability - result.market_price,
        result.confidence,
    )

    return result


async def screen_and_estimate_markets(
    markets: list[Market],
    conn: sqlite3.Connection,
    *,
    screen_model: str = SCREEN_MODEL,
    deep_model: str = DEFAULT_MODEL,
    max_searches: int = 3,
    ttl_hours: float = 12.0,
    max_concurrent: int = MAX_CONCURRENT,
    pre_threshold: float = 0.08,
) -> list[EstimationResult]:
    """Two-stage estimation: cheap Haiku screen, then Sonnet+search for divergent markets."""
    client = anthropic.Anthropic()
    sem = asyncio.Semaphore(max_concurrent)
    results: list[EstimationResult] = []

    # Stage 0: Check cache for deep results
    uncached_markets: list[Market] = []
    for market in markets:
        cached = get_cached_estimation(conn, market.id, ttl_hours)
        if cached:
            logger.info("Using cached estimation for: %s", market.question[:60])
            results.append(cached)
        else:
            uncached_markets.append(market)

    if not uncached_markets:
        logger.info("All %d markets served from cache", len(results))
        return results

    logger.info("Screening %d uncached markets with %s", len(uncached_markets), screen_model)

    # Stage 1: Screen all uncached markets with Haiku (no web search)
    async def _screen_one(market: Market) -> EstimationResult | None:
        async with sem:
            try:
                return await asyncio.to_thread(
                    screen_market, client, market, model=screen_model
                )
            except (anthropic.APIError, json.JSONDecodeError, ValueError) as e:
                logger.error("Failed to screen market %s: %s", market.id, e)
                return None

    screen_tasks = [_screen_one(m) for m in uncached_markets]
    screen_results = await asyncio.gather(*screen_tasks)

    # Stage 1 results: filter for deep analysis
    promote_markets: list[Market] = []
    for market, screen_result in zip(uncached_markets, screen_results):
        if screen_result is None:
            continue
        market_price = market.outcome_prices[0] if market.outcome_prices else 0.0
        divergence = abs(screen_result.model_probability - market_price)
        if divergence >= pre_threshold:
            promote_markets.append(market)
            logger.info(
                "Promoting to deep: %s (screen div: %.3f)", market.question[:50], divergence
            )
        else:
            results.append(screen_result)

    logger.info(
        "Screen complete: %d screened, %d promoted to deep analysis",
        len(uncached_markets), len(promote_markets),
    )

    if not promote_markets:
        return results

    # Stage 2: Deep estimation with Sonnet + web search
    async def _deep_one(market: Market) -> EstimationResult | None:
        async with sem:
            try:
                result = await asyncio.to_thread(
                    estimate_market, client, market, model=deep_model, max_searches=max_searches
                )
                insert_estimation(conn, result)
                return result
            except anthropic.RateLimitError as e:
                logger.warning("Rate limited estimating %s: %s", market.id, e)
                await asyncio.sleep(10)
                try:
                    result = await asyncio.to_thread(
                        estimate_market, client, market, model=deep_model, max_searches=max_searches
                    )
                    insert_estimation(conn, result)
                    return result
                except Exception:
                    logger.exception("Retry failed for market %s", market.id)
                    return None
            except (anthropic.APIError, json.JSONDecodeError, ValueError) as e:
                logger.error("Failed to estimate market %s: %s", market.id, e)
                return None

    deep_tasks = [_deep_one(m) for m in promote_markets]
    deep_results = await asyncio.gather(*deep_tasks)

    for r in deep_results:
        if r is not None:
            results.append(r)

    logger.info("Completed %d total estimations (%d deep)", len(results), len(promote_markets))
    return results


async def estimate_markets(
    markets: list[Market],
    conn: sqlite3.Connection,
    *,
    model: str = DEFAULT_MODEL,
    max_searches: int = 5,
    ttl_hours: float = 6.0,
    max_concurrent: int = MAX_CONCURRENT,
) -> list[EstimationResult]:
    """Estimate multiple markets with caching and bounded concurrency."""
    client = anthropic.Anthropic()
    sem = asyncio.Semaphore(max_concurrent)
    results: list[EstimationResult] = []

    async def _estimate_one(market: Market) -> EstimationResult | None:
        # Check cache first
        cached = get_cached_estimation(conn, market.id, ttl_hours)
        if cached:
            logger.info("Using cached estimation for: %s", market.question[:60])
            return cached

        async with sem:
            try:
                # Run sync Anthropic call in thread pool
                result = await asyncio.to_thread(
                    estimate_market, client, market, model=model, max_searches=max_searches
                )
                insert_estimation(conn, result)
                return result
            except anthropic.RateLimitError as e:
                logger.warning("Rate limited estimating %s: %s", market.id, e)
                await asyncio.sleep(10)
                try:
                    result = await asyncio.to_thread(
                        estimate_market, client, market, model=model, max_searches=max_searches
                    )
                    insert_estimation(conn, result)
                    return result
                except Exception:
                    logger.exception("Retry failed for market %s", market.id)
                    return None
            except (anthropic.APIError, json.JSONDecodeError, ValueError) as e:
                logger.error("Failed to estimate market %s: %s", market.id, e)
                return None

    tasks = [_estimate_one(m) for m in markets]
    task_results = await asyncio.gather(*tasks)

    for r in task_results:
        if r is not None:
            results.append(r)

    logger.info("Completed %d/%d estimations", len(results), len(markets))
    return results
