"""
run_experiment.py
------------------
Ties the whole pipeline together:
  1. Generate (or load) the synthetic multi-SKU / multi-warehouse demand data
  2. Evaluate forecast accuracy (MAPE)
  3. For every SKU/warehouse pair, run the baseline-vs-optimized rolling-
     horizon comparison via Monte Carlo simulation
  4. Aggregate into the three headline metrics: cost reduction %, service
     level %, and forecast MAPE %

Run directly:  python3 run_experiment.py
"""

import numpy as np
import pandas as pd

from generate_data import (
    generate_demand_history, SKU_UNIT_COST, HOLDING_COST_RATE,
    ORDERING_COST, STOCKOUT_PENALTY_MULT,
)
from forecasting import evaluate_forecast_accuracy
from rolling_horizon import evaluate_pair

TARGET_SERVICE_LEVEL = 0.72
SEED = 7


def main():
    rng = np.random.default_rng(SEED)

    print("Generating synthetic demand history...")
    df = generate_demand_history()

    print("Evaluating forecast accuracy (MAPE)...")
    acc = evaluate_forecast_accuracy(df)
    overall_mape = acc["mape"].mean()

    print("Running Monte Carlo baseline-vs-optimized simulation per SKU/warehouse...")
    results = []
    for (sku, wh), g in df.groupby(["sku", "warehouse"]):
        g = g.sort_values("date")
        demand_hist = g["demand"].values
        lt_mean = g["lead_time_mean"].iloc[0]
        lt_std = g["lead_time_std"].iloc[0]
        unit_cost = SKU_UNIT_COST[sku]

        res = evaluate_pair(
            demand_history=demand_hist, lead_time_mean=lt_mean, lead_time_std=lt_std,
            unit_cost=unit_cost, holding_cost_rate=HOLDING_COST_RATE,
            ordering_cost=ORDERING_COST, stockout_penalty_mult=STOCKOUT_PENALTY_MULT,
            target_service_level=TARGET_SERVICE_LEVEL, rng=rng,
        )
        res.update({"sku": sku, "warehouse": wh})
        results.append(res)

    res_df = pd.DataFrame(results)
    res_df.to_csv("/home/claude/inventory_project/data/results.csv", index=False)

    total_baseline_cost = res_df["baseline_cost"].sum()
    total_optimized_cost = res_df["optimized_cost"].sum()
    overall_cost_reduction = (total_baseline_cost - total_optimized_cost) / total_baseline_cost * 100
    overall_service_level = res_df["optimized_service_level"].mean()

    print("\n" + "=" * 55)
    print("HEADLINE RESULTS (multi-SKU, multi-warehouse)")
    print("=" * 55)
    print(f"Total baseline cost   : {total_baseline_cost:,.0f}")
    print(f"Total optimized cost  : {total_optimized_cost:,.0f}")
    print(f"Cost reduction        : {overall_cost_reduction:.1f}%")
    print(f"Service level (opt.)  : {overall_service_level * 100:.1f}%")
    print(f"Forecast MAPE         : {overall_mape:.1f}%")
    print("=" * 55)

    acc.to_csv("/home/claude/inventory_project/data/forecast_accuracy.csv", index=False)
    df.to_csv("/home/claude/inventory_project/data/synthetic_demand.csv", index=False)

    return res_df, acc, overall_cost_reduction, overall_service_level, overall_mape


if __name__ == "__main__":
    main()
