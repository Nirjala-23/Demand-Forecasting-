"""
rolling_horizon.py
-------------------
Compares two inventory policies over the SAME continuous evaluation
horizon, for every (SKU, warehouse) pair:

BASELINE  ("before" state): a naive, industry-common heuristic — hold a
    fixed number of days-of-cover based on trailing average demand, order
    in fixed lot sizes, and no statistically-derived safety stock. A
    single static (reorder_point, order_qty) is used for the whole
    horizon. This represents how these SKUs are typically managed before
    a formal inventory-optimization project.

OPTIMIZED ("after" state): a rolling-horizon policy that re-forecasts
    demand every REPLAN_EVERY days from a trailing window, recomputes the
    reorder point / safety stock (targeting the desired service level)
    and EOQ order quantity via policy.compute_policy(), and hands the
    updated (reorder_point, order_qty) to the SAME continuous simulation
    — inventory and in-transit orders carry across re-planning
    boundaries, so re-planning doesn't artificially reset the system.

Both policies are evaluated with Monte Carlo simulation
(simulation.simulate_schedule) over the same horizon length, so the
comparison isolates the effect of the *policy*, not random noise.
"""

import numpy as np

from policy import compute_policy
from simulation import simulate_schedule

REPLAN_EVERY = 21          # days between re-forecast/re-optimization
EVAL_HORIZON = 270         # total days evaluated (baseline and optimized)
N_REPS = 150               # Monte Carlo replications per pair
WARM_START = 30            # burn-in days excluded from scoring

BASELINE_LOT_DAYS = 20             # naive fixed order size: ~3 weeks of avg demand
# The naive baseline reorders at exactly the expected lead-time demand with
# NO safety-stock buffer at all — a common real-world mistake (sizing stock
# to the "average case" and ignoring demand/lead-time variability), which is
# why it runs into stockouts on roughly half of replenishment cycles.


def _series_stats(values: np.ndarray):
    return float(np.mean(values)), float(np.std(values) + 1e-6)


def evaluate_pair(
    demand_history: np.ndarray, lead_time_mean: float, lead_time_std: float,
    unit_cost: float, holding_cost_rate: float, ordering_cost: float,
    stockout_penalty_mult: float, target_service_level: float,
    rng: np.random.Generator,
) -> dict:
    train = demand_history[:-EVAL_HORIZON]
    eval_pool = demand_history[-EVAL_HORIZON:]   # bootstrap pool drawn from held-out period
    d_mean_full, d_std_full = _series_stats(train)

    # ---------- BASELINE: static, naive "cover the average" policy ----------
    baseline_rop = lead_time_mean * d_mean_full        # no safety-stock term at all
    baseline_qty = BASELINE_LOT_DAYS * d_mean_full
    baseline_schedule = [{"start_day": 0, "end_day": EVAL_HORIZON,
                           "reorder_point": baseline_rop, "order_qty": baseline_qty}]
    baseline_res = simulate_schedule(
        demand_samples_by_day=eval_pool, schedule=baseline_schedule,
        lead_time_mean=lead_time_mean, lead_time_std=lead_time_std, unit_cost=unit_cost,
        holding_cost_rate=holding_cost_rate, ordering_cost=ordering_cost,
        stockout_penalty_mult=stockout_penalty_mult, horizon_days=EVAL_HORIZON,
        n_reps=N_REPS, rng=rng, warm_start_days=WARM_START,
    )

    # ---------- OPTIMIZED: rolling-horizon, re-planned policy ----------
    schedule = []
    n_periods = int(np.ceil(EVAL_HORIZON / REPLAN_EVERY))
    for p in range(n_periods):
        start_day = p * REPLAN_EVERY
        end_day = min(EVAL_HORIZON, start_day + REPLAN_EVERY)
        # trailing 150-day window ending just before this period starts, using
        # the full history (train + already-elapsed eval days) as if it were
        # newly observed actuals feeding the next re-forecast
        history_so_far = demand_history[: len(train) + start_day]
        window = history_so_far[-150:]
        d_mean, d_std = _series_stats(window)
        annual_demand = d_mean * 365

        params = compute_policy(
            d_mean=d_mean, d_std=d_std, lt_mean=lead_time_mean, lt_std=lead_time_std,
            annual_demand=annual_demand, unit_cost=unit_cost, ordering_cost=ordering_cost,
            holding_cost_rate=holding_cost_rate, service_level=target_service_level,
        )
        schedule.append({"start_day": start_day, "end_day": end_day,
                          "reorder_point": params.reorder_point, "order_qty": params.order_quantity})

    optimized_res = simulate_schedule(
        demand_samples_by_day=eval_pool, schedule=schedule,
        lead_time_mean=lead_time_mean, lead_time_std=lead_time_std, unit_cost=unit_cost,
        holding_cost_rate=holding_cost_rate, ordering_cost=ordering_cost,
        stockout_penalty_mult=stockout_penalty_mult, horizon_days=EVAL_HORIZON,
        n_reps=N_REPS, rng=rng, warm_start_days=WARM_START,
    )

    baseline_total_cost = baseline_res.total_cost
    optimized_total_cost = optimized_res.total_cost

    return {
        "baseline_cost": baseline_total_cost,
        "baseline_service_level": baseline_res.service_level,
        "optimized_cost": optimized_total_cost,
        "optimized_service_level": optimized_res.service_level,
        "cost_reduction_pct": (baseline_total_cost - optimized_total_cost) / baseline_total_cost * 100,
    }
