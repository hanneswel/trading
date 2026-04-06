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


@dataclass
class EstimationResult:
    market_id: str
    market_question: str
    model_probability: float  # 0.0 to 1.0, for the "Yes" outcome
    confidence: str  # "low", "medium", "high"
    reasoning: str
    factors_for: list[str]
    factors_against: list[str]
    sources: list[str]  # URLs from web search
    market_price: float  # current market implied prob at time of estimation
    model_name: str
    input_tokens: int
    output_tokens: int
    search_count: int
    stage: str = "deep"  # "screen" or "deep"
    estimated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Signal:
    market_id: str
    market_question: str
    market_price: float  # current "Yes" price
    model_probability: float
    divergence: float  # model_prob - market_price (positive = model thinks underpriced)
    confidence: str
    reasoning: str
    expected_value: float  # abs(divergence) * payout (simplified)
    side: str  # "BUY_YES" or "BUY_NO"
    estimated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
