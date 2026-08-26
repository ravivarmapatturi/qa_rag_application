"""Turn token-usage counts and wall-clock timing into $/query estimates and
latency percentiles for the eval report.

Doesn't capture usage data itself -- that comes from
observability.usage_tracking_config() (a LangChain UsageMetadataCallbackHandler
attached alongside the Langfuse tracing callback, piggybacking on the same
callbacks list rather than being separate instrumentation) or from Langfuse's
own dashboard, which captures the same per-call usage automatically whenever
tracing is enabled. This module is just arithmetic on top of that.
"""
import time
from contextlib import contextmanager
from typing import Iterator, Optional

# Approximate USD price per 1 million tokens, (input, output). Update as
# provider pricing changes -- covers the two models query_translation.get_llm
# can build. Unknown models return None from estimate_cost_usd rather than a
# silently wrong number.
MODEL_PRICING_PER_MILLION_TOKENS = {
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gpt-3.5-turbo": (0.50, 1.50),
}


class Timer:
    """`with Timer() as t: ...` then t.elapsed_seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_seconds: Optional[float] = None
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start


@contextmanager
def timed() -> Iterator[Timer]:
    t = Timer()
    with t:
        yield t


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model_name)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def total_tokens_and_cost(usage_metadata: dict) -> dict:
    """Sums a UsageMetadataCallbackHandler's `usage_metadata` (one entry per
    model actually called during the run -- an agentic run can call the same
    model many times across routing/grading/generation, all aggregated under
    one key) into a single {input_tokens, output_tokens, total_tokens,
    cost_usd} record. cost_usd is None if no model in the run has a known
    price rather than silently treating unpriced usage as free.
    """
    input_tokens = sum(v.get("input_tokens", 0) for v in usage_metadata.values())
    output_tokens = sum(v.get("output_tokens", 0) for v in usage_metadata.values())
    total_tokens = sum(v.get("total_tokens", 0) for v in usage_metadata.values())

    costs = [
        estimate_cost_usd(model, v.get("input_tokens", 0), v.get("output_tokens", 0))
        for model, v in usage_metadata.items()
    ]
    known_costs = [c for c in costs if c is not None]
    cost_usd = sum(known_costs) if known_costs else None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def percentile(values: list, pct: float) -> Optional[float]:
    """Linear-interpolation percentile (pct in [0, 100]) -- e.g. percentile(values, 95) for p95."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def latency_summary(latencies_seconds: list) -> dict:
    return {
        "p50_seconds": percentile(latencies_seconds, 50),
        "p95_seconds": percentile(latencies_seconds, 95),
        "mean_seconds": sum(latencies_seconds) / len(latencies_seconds) if latencies_seconds else None,
        "max_seconds": max(latencies_seconds) if latencies_seconds else None,
    }
