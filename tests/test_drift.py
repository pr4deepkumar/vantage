import pytest
from vantage.eval.drift import calculate_mean, calculate_std, calculate_psi, check_for_drift

def test_calculate_mean():
    assert calculate_mean([10.0, 20.0, 30.0]) == 20.0
    assert calculate_mean([]) == 0.0

def test_calculate_std():
    vals = [10.0, 20.0, 30.0]
    mean = calculate_mean(vals)
    std = calculate_std(vals, mean)
    assert round(std, 2) == 10.0
    assert calculate_std([5.0], 5.0) == 0.0

def test_calculate_psi():
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    target_same = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    target_drifted = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
    
    psi_same = calculate_psi(baseline, target_same)
    psi_drift = calculate_psi(baseline, target_drifted)
    
    assert psi_same < 0.1
    assert psi_drift > 0.2

def test_check_for_drift_detection():
    # Build 40 historical normal traces
    traces = []
    for i in range(40):
        traces.append({
            "timestamp": f"2026-07-29T10:{i:02d}:00",
            "status": "success",
            "latency_ms": 100.0,
            "cost": 0.001,
            "output_tokens": 50
        })
    
    # Add 30 recent window traces with severe latency & token drift
    for i in range(30):
        traces.append({
            "timestamp": f"2026-07-29T11:{i:02d}:00",
            "status": "success",
            "latency_ms": 500.0, # 5x latency jump
            "cost": 0.005,
            "output_tokens": 250
        })
        
    alerts = check_for_drift(traces, window_size=30, threshold_std=2.0)
    assert len(alerts) >= 1
    alert_metrics = [a["metric"] for a in alerts]
    assert "latency_ms" in alert_metrics
