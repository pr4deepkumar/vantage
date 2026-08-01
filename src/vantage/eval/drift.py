import math
import logging

logger = logging.getLogger("vantage.eval.drift")

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
    eps = 1e-4 # Epsilon to prevent division by zero or log(0)
    
    for i in range(num_buckets):
        b_lower = min_val + i * step
        b_upper = min_val + (i + 1) * step
        
        b_count = sum(1 for x in baseline if b_lower <= x < b_upper or (i == num_buckets - 1 and x == b_upper))
        t_count = sum(1 for x in target if b_lower <= x < b_upper or (i == num_buckets - 1 and x == b_upper))
        
        b_prop = (b_count / len(baseline)) + eps
        t_prop = (t_count / len(target)) + eps
        
        psi_value += (t_prop - b_prop) * math.log(t_prop / b_prop)
        
    return psi_value

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
    sorted_traces = sorted(traces, key=lambda t: t.get("timestamp", ""))
    successful_traces = [t for t in sorted_traces if t.get("status") == "success"]
    total_count = len(successful_traces)
    
    min_required = window_size * 2
    if total_count < min_required:
        logger.info(f"Insufficient trace volume for drift detection ({total_count}/{min_required} required).")
        return []
        
    window_traces = successful_traces[-window_size:]
    baseline_traces = successful_traces[:-window_size]
    
    metrics = {
        "latency_ms": {
            "display_name": "Latency (ms)",
            "min_std": 20.0,
        },
        "cost": {
            "display_name": "USD Cost ($)",
            "min_std": 0.0001,
        },
        "output_tokens": {
            "display_name": "Output Token Length",
            "min_std": 5.0,
        }
    }
    
    alerts = []
    
    for metric_key, config in metrics.items():
        baseline_vals = [t.get(metric_key, 0.0) for t in baseline_traces if t.get(metric_key) is not None]
        window_vals = [t.get(metric_key, 0.0) for t in window_traces if t.get(metric_key) is not None]
        
        if not baseline_vals or not window_vals:
            continue
            
        b_mean = calculate_mean(baseline_vals)
        b_std = calculate_std(baseline_vals, b_mean)
        w_mean = calculate_mean(window_vals)
        
        effective_std = max(b_std, config["min_std"])
        
        diff = w_mean - b_mean
        z_score = abs(diff) / effective_std
        
        if z_score > threshold_std:
            direction = "increased" if diff > 0 else "decreased"
            alerts.append({
                "metric": metric_key,
                "display_name": config["display_name"],
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "window_mean": w_mean,
                "z_score": z_score,
                "direction": direction,
                "message": f"Drift Detected in {config['display_name']}! "
                           f"Current window average of {w_mean:.4f} is {z_score:.2f} std devs away from the baseline of {b_mean:.4f}."
            })
            
    return alerts
