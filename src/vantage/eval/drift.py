import logging
import math

logger = logging.getLogger("vantage.eval.drift")

# Metrics compared between the baseline and the latest window. ``min_std`` keeps a near-constant
# baseline from turning tiny wobbles into huge z-scores.
METRICS = {
    "latency_ms": {"display_name": "Latency (ms)", "min_std": 20.0},
    "cost": {"display_name": "USD Cost ($)", "min_std": 0.0001},
    "output_tokens": {"display_name": "Output Token Length", "min_std": 5.0},
}


def calculate_mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_std(values: list, mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_psi(baseline: list, target: list, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and target distributions.
    PSI < 0.1: No change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant drift
    """
    if not baseline or not target:
        return 0.0

    min_val = min(min(baseline), min(target))
    max_val = max(max(baseline), max(target))
    if min_val == max_val:
        return 0.0

    step = (max_val - min_val) / num_buckets
    psi_value = 0.0
    eps = 1e-4  # Epsilon to prevent division by zero or log(0)

    for i in range(num_buckets):
        b_lower = min_val + i * step
        b_upper = min_val + (i + 1) * step

        b_count = sum(
            1 for x in baseline if b_lower <= x < b_upper or (i == num_buckets - 1 and x == b_upper)
        )
        t_count = sum(1 for x in target if b_lower <= x < b_upper or (i == num_buckets - 1 and x == b_upper))

        b_prop = (b_count / len(baseline)) + eps
        t_prop = (t_count / len(target)) + eps

        psi_value += (t_prop - b_prop) * math.log(t_prop / b_prop)

    return psi_value


def compare_windows(traces: list, window_size: int = 30, threshold_std: float = 2.0) -> list:
    """
    Compares the mean of the last ``window_size`` successful calls with the baseline (every
    successful call before them), for each metric in ``METRICS``. Returns one comparison per
    metric, with ``drifted`` set when the window mean is more than ``threshold_std`` baseline
    standard deviations away. Empty until there are at least ``2 * window_size`` calls.
    """
    sorted_traces = sorted(traces, key=lambda t: t.get("timestamp", ""))
    successful_traces = [t for t in sorted_traces if t.get("status") == "success"]
    total_count = len(successful_traces)

    min_required = window_size * 2
    if total_count < min_required:
        logger.info(f"Insufficient trace volume for drift detection ({total_count}/{min_required} required).")
        return []

    window_traces = successful_traces[-window_size:]
    baseline_traces = successful_traces[:-window_size]

    comparisons = []
    for metric_key, config in METRICS.items():
        baseline_vals = [t[metric_key] for t in baseline_traces if t.get(metric_key) is not None]
        window_vals = [t[metric_key] for t in window_traces if t.get(metric_key) is not None]
        if not baseline_vals or not window_vals:
            continue

        b_mean = calculate_mean(baseline_vals)
        b_std = calculate_std(baseline_vals, b_mean)
        w_mean = calculate_mean(window_vals)
        diff = w_mean - b_mean
        z_score = abs(diff) / max(b_std, config["min_std"])
        comparisons.append(
            {
                "metric": metric_key,
                "display_name": config["display_name"],
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "window_mean": w_mean,
                "z_score": z_score,
                "direction": "increased" if diff > 0 else "decreased",
                "drifted": z_score > threshold_std,
            }
        )
    return comparisons


def check_for_drift(traces: list, window_size: int = 30, threshold_std: float = 2.0) -> list:
    """
    Computes rolling averages for the last N requests (current window) and
    compares them to historical baselines (all requests prior to the current window).

    Metrics analyzed:
    - Latency (latency_ms)
    - Cost (cost)
    - Output length (output_tokens)

    Returns a list of alerts.
    """
    return [
        comparison
        | {
            "message": f"{comparison['display_name']} drifted: the latest window averages "
            f"{comparison['window_mean']:.4f}, {comparison['z_score']:.2f} standard deviations from the "
            f"baseline of {comparison['baseline_mean']:.4f}."
        }
        for comparison in compare_windows(traces, window_size, threshold_std)
        if comparison["drifted"]
    ]
