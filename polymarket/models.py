from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Market:
    id: str
    question: str
    condition_id: str
    slug: str
    description: str
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str]
    volume: float
    volume_24hr: float
    liquidity: float
    active: bool
    closed: bool
    end_date: str | None = None
    spread: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    last_trade_price: float | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OrderLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    market_condition_id: str
    asset_id: str
    outcome: str
    bids: list[OrderLevel]
    asks: list[OrderLevel]
    midpoint: float | None = None
    spread: float | None = None
    last_trade_price: float | None = None
    timestamp: str | None = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
