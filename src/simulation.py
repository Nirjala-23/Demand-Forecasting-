"""
simulation.py
-------------
Monte Carlo simulation of a single-item, single-location (s, Q) inventory
system: whenever the inventory position drops to/below the reorder point
`s`, place an order of size `Q`, which arrives after a stochastic lead
time. Demand each day is drawn (bootstrap) from the empirical daily-
demand distribution so the simulation reflects the *actual* demand
variability in the data, not just a normal assumption.

The policy is expressed as a *schedule* of (reorder_point, order_qty)
segments over the simulated horizon. This is what makes the rolling-
horizon comparison fair and realistic: instead of resetting inventory to
"freshly full" every time the policy is re-planned (which would trivially
inflate the service level), the simulation carries on-hand inventory and
in-transit orders continuously across re-planning boundaries and simply
starts applying the newly re-optimized (s, Q) from that day onward. A
static (baseline) policy is just a schedule with a single segment.

Run many independent replications (Monte Carlo) and average the results
to get stable estimates of:
  - service level (fraction of demand met directly from stock)
  - average on-hand inventory (drives holding cost)
  - number of orders placed (drives ordering cost)
  - unmet ("stocked out") demand units (drives stockout penalty cost)
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class SimResult:
    service_level: float
    avg_on_hand: float
    n_orders: float
    stockout_units: float
    total_cost: float


def _policy_at(schedule: List[dict], day: int) -> Tuple[float, float]:
    for seg in schedule:
        if seg["start_day"] <= day < seg["end_day"]:
            return seg["reorder_point"], seg["order_qty"]
    last = schedule[-1]
    return last["reorder_point"], last["order_qty"]


def simulate_schedule(
    demand_samples_by_day, schedule: List[dict], lead_time_mean: float, lead_time_std: float,
    unit_cost: float, holding_cost_rate: float, ordering_cost: float,
    stockout_penalty_mult: float, horizon_days: int, n_reps: int = 150,
    rng: np.random.Generator = None, warm_start_days: int = 30,
) -> SimResult:
    """
    demand_samples_by_day: either a single 1-D array (bootstrap pool used
    for every day) or a list of per-day bootstrap pools of the same length
    as horizon_days (so the demand distribution can itself evolve, e.g.
    trailing windows for later periods).
    """
    rng = rng or np.random.default_rng()
    holding_cost_per_unit_day = unit_cost * holding_cost_rate / 365
    stockout_cost_per_unit = unit_cost * stockout_penalty_mult

    def pool_for_day(d):
        if isinstance(demand_samples_by_day, list):
            return demand_samples_by_day[min(d, len(demand_samples_by_day) - 1)]
        return demand_samples_by_day

    service_levels, on_hand_avgs, order_counts, stockout_totals, total_costs = [], [], [], [], []

    for _ in range(n_reps):
        rop0, q0 = _policy_at(schedule, 0)
        on_hand = rop0 + q0 * 0.5   # realistic mid-cycle starting stock, not "freshly full"
        pending_orders = []
        cum_on_hand = 0.0
        demand_met = 0.0
        demand_total = 0.0
        stockout_units = 0.0
        n_orders = 0
        holding_cost = 0.0
        ordering_cost_total = 0.0

        for day in range(horizon_days):
            rop, qty = _policy_at(schedule, day)

            arrivals = [q for arr_day, q in pending_orders if arr_day == day]
            for q in arrivals:
                on_hand += q
            pending_orders = [(a, q) for a, q in pending_orders if a != day]

            d = rng.choice(pool_for_day(day))
            score_day = day >= warm_start_days

            shipped = min(on_hand, d)
            shortfall = d - shipped
            on_hand -= shipped

            if score_day:
                demand_total += d
                demand_met += shipped
                stockout_units += shortfall

            cum_on_hand += on_hand
            holding_cost += on_hand * holding_cost_per_unit_day

            inventory_position = on_hand + sum(q for _, q in pending_orders)
            if inventory_position <= rop:
                lt = max(1, int(round(rng.normal(lead_time_mean, lead_time_std))))
                pending_orders.append((day + lt, qty))
                n_orders += 1
                ordering_cost_total += ordering_cost

        stockout_cost_total = stockout_units * stockout_cost_per_unit
        total_cost = holding_cost + ordering_cost_total + stockout_cost_total

        service_levels.append(demand_met / max(demand_total, 1e-6))
        on_hand_avgs.append(cum_on_hand / horizon_days)
        order_counts.append(n_orders)
        stockout_totals.append(stockout_units)
        total_costs.append(total_cost)

    return SimResult(
        service_level=float(np.mean(service_levels)),
        avg_on_hand=float(np.mean(on_hand_avgs)),
        n_orders=float(np.mean(order_counts)),
        stockout_units=float(np.mean(stockout_totals)),
        total_cost=float(np.mean(total_costs)),
    )
