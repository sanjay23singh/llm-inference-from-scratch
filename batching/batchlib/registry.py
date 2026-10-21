"""One place to build any policy by name, so experiment scripts stay short."""
from . import paths  # noqa: F401
from step2_static.policy import StaticBatching  # noqa: E402
from step3_dynamic.policy import DynamicBatching  # noqa: E402
from step4_continuous.policy import ContinuousBatching  # noqa: E402
from step6_chunked_prefill.policy import ChunkedPrefill  # noqa: E402


def make_policy(name, n_slots, max_wait=0.5, token_budget=128):
    """n_slots doubles as the static/dynamic batch size so every policy gets the same concurrency."""
    return {
        "static": lambda: StaticBatching(n_slots),
        "dynamic": lambda: DynamicBatching(n_slots, max_wait),
        "bucketed": lambda: DynamicBatching(n_slots, max_wait, length_aware=True),
        "continuous": lambda: ContinuousBatching("prefill_first"),
        "continuous_mixed": lambda: ContinuousBatching("mixed"),
        "chunked": lambda: ChunkedPrefill(token_budget),
    }[name]()


ALL = ["static", "dynamic", "bucketed", "continuous", "continuous_mixed", "chunked"]
