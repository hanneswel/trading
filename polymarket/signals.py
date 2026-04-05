from .models import EstimationResult, Signal

DEFAULT_THRESHOLD = 0.10


def detect_signal(
    estimation: EstimationResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> Signal | None:
    """Compare model estimate to market price and return a Signal if divergence exceeds threshold."""
    divergence = estimation.model_probability - estimation.market_price

    if abs(divergence) < threshold:
        return None

    # Determine trade direction
    if divergence > 0:
        # Model thinks Yes is underpriced — buy Yes
        side = "BUY_YES"
        # If we buy Yes at market_price and it resolves Yes, payout is 1.0
        edge = divergence
        cost = estimation.market_price
    else:
        # Model thinks Yes is overpriced — buy No (equivalent to selling Yes)
        side = "BUY_NO"
        edge = abs(divergence)
        cost = 1.0 - estimation.market_price

    # Expected value: edge * (payout / cost) simplified
    # Buying at `cost`, winning `1.0`, so profit = 1.0 - cost
    # EV = P(win) * profit - P(lose) * cost
    # where P(win) = model_probability for BUY_YES, 1-model_probability for BUY_NO
    if side == "BUY_YES":
        p_win = estimation.model_probability
        ev = p_win * (1.0 - cost) - (1.0 - p_win) * cost
    else:
        p_win = 1.0 - estimation.model_probability
        ev = p_win * (1.0 - cost) - (1.0 - p_win) * cost

    return Signal(
        market_id=estimation.market_id,
        market_question=estimation.market_question,
        market_price=estimation.market_price,
        model_probability=estimation.model_probability,
        divergence=divergence,
        confidence=estimation.confidence,
        reasoning=estimation.reasoning,
        expected_value=ev,
        side=side,
        estimated_at=estimation.estimated_at,
    )


def detect_signals(
    estimations: list[EstimationResult],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Signal]:
    """Detect signals from a list of estimations, sorted by expected value descending."""
    signals = []
    for est in estimations:
        sig = detect_signal(est, threshold)
        if sig is not None:
            signals.append(sig)

    signals.sort(key=lambda s: s.expected_value, reverse=True)
    return signals
