"""
generate_data.py
-----------------
Creates a synthetic daily demand history for multiple SKUs across multiple
warehouses. Each (SKU, warehouse) pair gets its own base demand level,
trend, weekly + monthly seasonality, and noise, plus a warehouse-specific
supplier lead time (with day-to-day variability). This mimics a real
multi-SKU / multi-warehouse supply chain dataset well enough to test
forecasting + inventory policies against.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SKUS = [f"SKU-{i:02d}" for i in range(1, 6)]          # 5 SKUs
WAREHOUSES = ["WH-North", "WH-Central", "WH-South"]     # 3 warehouses
N_DAYS = 730                                            # 2 years of daily data

# Base demand + lead time characteristics per SKU (units/day) and per warehouse
SKU_BASE_DEMAND = {"SKU-01": 120, "SKU-02": 60, "SKU-03": 200, "SKU-04": 35, "SKU-05": 90}
SKU_TREND = {"SKU-01": 0.03, "SKU-02": -0.01, "SKU-03": 0.05, "SKU-04": 0.00, "SKU-05": 0.02}
SKU_NOISE_STD_PCT = {"SKU-01": 0.24, "SKU-02": 0.28, "SKU-03": 0.20, "SKU-04": 0.36, "SKU-05": 0.26}

WH_DEMAND_SHARE = {"WH-North": 0.45, "WH-Central": 0.35, "WH-South": 0.20}
WH_BASE_LEAD_TIME = {"WH-North": 5, "WH-Central": 7, "WH-South": 10}   # days
WH_LEAD_TIME_STD = {"WH-North": 1.0, "WH-Central": 1.5, "WH-South": 2.5}

# Unit economics (used later by the cost model)
SKU_UNIT_COST = {"SKU-01": 18, "SKU-02": 45, "SKU-03": 9, "SKU-04": 70, "SKU-05": 25}
HOLDING_COST_RATE = 0.22        # annual holding cost as a fraction of unit cost
ORDERING_COST = 120             # fixed cost per purchase order placed
STOCKOUT_PENALTY_MULT = 3.0     # stockout cost per unit short, as a multiple of unit cost


def _seasonality(day_index: np.ndarray) -> np.ndarray:
    weekly = 1 + 0.15 * np.sin(2 * np.pi * day_index / 7)
    monthly = 1 + 0.10 * np.sin(2 * np.pi * day_index / 30.4 + 1.2)
    return weekly * monthly


def generate_demand_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    day_index = np.arange(N_DAYS)
    rows = []

    for sku in SKUS:
        base = SKU_BASE_DEMAND[sku]
        trend = SKU_TREND[sku]
        noise_pct = SKU_NOISE_STD_PCT[sku]
        seasonal = _seasonality(day_index)
        trended = base * (1 + trend * day_index / 365)
        mean_demand = trended * seasonal

        for wh in WAREHOUSES:
            share = WH_DEMAND_SHARE[wh]
            wh_mean = mean_demand * share
            noise = RNG.normal(0, noise_pct * wh_mean, size=N_DAYS)
            demand = np.clip(wh_mean + noise, 0, None)
            demand = RNG.poisson(demand).astype(float)  # discretize, add count noise

            lt_mean = WH_BASE_LEAD_TIME[wh]
            lt_std = WH_LEAD_TIME_STD[wh]

            for i, d in enumerate(dates):
                rows.append({
                    "date": d, "sku": sku, "warehouse": wh, "demand": demand[i],
                    "lead_time_mean": lt_mean, "lead_time_std": lt_std,
                })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_demand_history()
    df.to_csv("/home/claude/inventory_project/data/synthetic_demand.csv", index=False)
    print(df.shape)
    print(df.groupby(["sku", "warehouse"])["demand"].mean().round(1))
