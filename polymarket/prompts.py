from datetime import datetime, timezone

from .models import Market

SYSTEM_PROMPT = """\
You are a calibrated probability estimator for prediction markets. Your job is to \
estimate the true probability of events resolving "Yes", given all available evidence.

CALIBRATION RULES — follow these strictly:
1. DO NOT anchor on the current market price. Assess the evidence independently first, \
then compare to the market price only at the end.
2. DO NOT round to convenient numbers (0.5, 0.7, 0.9, etc.). Use precise values like \
0.63 or 0.41. Real-world probabilities are rarely round.
3. DO NOT be overconfident. If information is sparse or conflicting, your probability \
should reflect genuine uncertainty (closer to 0.5). A probability of 0.95+ requires \
overwhelming evidence.
4. DO NOT anchor on the question framing. "Will X happen?" does not make X more likely. \
Consider the base rate of similar events.
5. Consider base rates first, then update based on specific evidence. This is Bayesian \
reasoning.
6. For events far in the future, bias toward base rates and uncertainty.
7. For events that have nearly resolved (deadline tomorrow, outcome almost certain), \
you can use extreme probabilities.
8. Account for the possibility that you are wrong or missing information. Even when \
evidence strongly points one way, leave room for surprise.

SEARCH STRATEGY:
- Search for the most recent news and data about the topic.
- For political markets: look for recent polls, expert analysis, and precedent.
- For economic markets: look for consensus forecasts, leading indicators, and recent data releases.
- For sports markets: look for recent results, injuries, and betting odds from sportsbooks.
- For crypto/financial markets: look for recent price action, on-chain data, and analyst forecasts.
- Prioritize sources from the last 7 days. Older information may be stale.

OUTPUT FORMAT:
You must respond with a JSON object matching this exact schema:
{
    "factors_for": ["reason 1 supporting Yes", "reason 2 supporting Yes"],
    "factors_against": ["reason 1 supporting No", "reason 2 supporting No"],
    "base_rate_reasoning": "What is the historical base rate for this type of event?",
    "probability": 0.XX,
    "confidence": "low|medium|high",
    "reasoning": "2-3 sentence summary of your overall assessment"
}

CONFIDENCE LEVELS:
- "low": Limited or conflicting information available; estimate is largely based on priors/base rates
- "medium": Some relevant evidence found; estimate is informed but significant uncertainty remains
- "high": Strong, recent, consistent evidence available; estimate is well-supported

Return ONLY the JSON object, no other text.\
"""


def build_user_prompt(market: Market, current_date: datetime | None = None) -> str:
    if current_date is None:
        current_date = datetime.now(timezone.utc)

    date_str = current_date.strftime("%Y-%m-%d")
    yes_price = market.outcome_prices[0] if market.outcome_prices else "unknown"
    no_price = market.outcome_prices[1] if len(market.outcome_prices) > 1 else "unknown"

    description = market.description or "No additional description provided."
    # Truncate very long descriptions
    if len(description) > 2000:
        description = description[:2000] + "..."

    end_date_str = ""
    if market.end_date:
        end_date_str = f"\nResolution deadline: {market.end_date}"

    return f"""\
Estimate the probability that the following prediction market resolves "Yes".

MARKET QUESTION: {market.question}

DESCRIPTION / RESOLUTION CRITERIA:
{description}
{end_date_str}
TODAY'S DATE: {date_str}

MARKET DATA (for context only — do NOT anchor on these prices):
- Current "Yes" price: {yes_price}
- Current "No" price: {no_price}
- 24h volume: ${market.volume_24hr:,.0f}
- Total volume: ${market.volume:,.0f}

Search for recent, relevant information about this topic, then provide your calibrated \
probability estimate as a JSON object.\
"""


# JSON schema for structured output via Anthropic API
ESTIMATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "factors_for": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key factors supporting a 'Yes' resolution",
        },
        "factors_against": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key factors supporting a 'No' resolution",
        },
        "base_rate_reasoning": {
            "type": "string",
            "description": "Historical base rate analysis for this type of event",
        },
        "probability": {
            "type": "number",
            "description": "Estimated probability of 'Yes' resolution (0.0 to 1.0)",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence level in the estimate",
        },
        "reasoning": {
            "type": "string",
            "description": "2-3 sentence summary of the overall assessment",
        },
    },
    "required": [
        "factors_for",
        "factors_against",
        "base_rate_reasoning",
        "probability",
        "confidence",
        "reasoning",
    ],
    "additionalProperties": False,
}
