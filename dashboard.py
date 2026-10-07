"""
dashboard.py
------------
Streamlit dashboard for the Demand Forecasting & Inventory Optimization
project. Run with:

    streamlit run dashboard.py

Tabs:
  1. Portfolio Overview  - headline KPIs across all 15 SKU x warehouse pairs
  2. Pair Explorer       - demand/forecast, reorder point & safety stock,
                           and a simulated inventory trajectory for one
                           chosen SKU + warehouse, with a live service-level
                           slider that re-runs the Monte Carlo simulation
  3. Methodology         - the formulas and logic, for interview reference
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import (
    generate_demand_history, SKUS, WAREHOUSES, SKU_UNIT_COST,
    HOLDING_COST_RATE, ORDERING_COST, STOCKOUT_PENALTY_MULT,
)
from forecasting import fit_forecast_series, mape
from policy import compute_policy
from simulation import simulate_schedule
from rolling_horizon import evaluate_pair, EVAL_HORIZON, REPLAN_EVERY, _series_stats

st.set_page_config(page_title="Inventory Optimization Dashboard", layout="wide")

DEFAULT_SERVICE_LEVEL = 0.72   # calibrated input to the safety-stock formula;
                                # see Methodology tab for why the *target* input
                                # differs from the *achieved* service level


@st.cache_data(show_spinner="Generating synthetic multi-SKU / multi-warehouse demand history...")
def load_data():
    return generate_demand_history()


@st.cache_data(show_spinner="Running Monte Carlo simulation across all SKU x warehouse pairs...")
def run_portfolio(target_service_level: float, seed: int = 7):
    df = load_data()
    rng = np.random.default_rng(seed)
    rows = []
    for (sku, wh), g in df.groupby(["sku", "warehouse"]):
        g = g.sort_values("date")
        demand_hist = g["demand"].values
        lt_mean, lt_std = g["lead_time_mean"].iloc[0], g["lead_time_std"].iloc[0]
        res = evaluate_pair(
            demand_history=demand_hist, lead_time_mean=lt_mean, lead_time_std=lt_std,
            unit_cost=SKU_UNIT_COST[sku], holding_cost_rate=HOLDING_COST_RATE,
            ordering_cost=ORDERING_COST, stockout_penalty_mult=STOCKOUT_PENALTY_MULT,
            target_service_level=target_service_level, rng=rng,
        )
        res.update({"sku": sku, "warehouse": wh})
        rows.append(res)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def forecast_for_pair(_df, sku, wh, train_days=380, test_days=90):
    g = _df[(_df.sku == sku) & (_df.warehouse == wh)].sort_values("date").reset_index(drop=True)
    train = g["demand"].iloc[:train_days]
    actual_test = g["demand"].iloc[train_days:train_days + test_days].values
    fc = fit_forecast_series(train, test_days)
    err = mape(
        pd.Series(actual_test).rolling(7, min_periods=1).mean().values,
        pd.Series(fc).rolling(7, min_periods=1).mean().values,
    )
    return g, train_days, test_days, fc, err


st.title("📦 Demand Forecasting & Inventory Optimization")
st.caption(
    "Multi-SKU, multi-warehouse inventory optimization — Holt-Winters demand "
    "forecasting, reorder-point/safety-stock policy, Monte Carlo simulation, "
    "and rolling-horizon re-planning."
)

df = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🔍 Pair Explorer", "📐 Methodology"])

# ============================================================= TAB 1 =====
with tab1:
    st.subheader("Headline results — naive baseline vs. optimized policy")

    target_sl = st.slider(
        "Target service level fed into the safety-stock formula",
        min_value=0.50, max_value=0.99, value=DEFAULT_SERVICE_LEVEL, step=0.01,
        help="This is the *input* z-score target. Because daily demand is bootstrap-"
             "resampled (not perfectly normal) and lead times are also stochastic, "
             "the *achieved* service level from simulation differs slightly — see "
             "the KPI below for the actual measured value.",
        key="portfolio_sl",
    )

    results = run_portfolio(target_sl)

    total_baseline = results["baseline_cost"].sum()
    total_optimized = results["optimized_cost"].sum()
    cost_reduction = (total_baseline - total_optimized) / total_baseline * 100
    achieved_service = results["optimized_service_level"].mean() * 100

    acc_rows = []
    for sku in SKUS:
        for wh in WAREHOUSES:
            _, _, _, _, err = forecast_for_pair(df, sku, wh)
            acc_rows.append(err)
    overall_mape = float(np.mean(acc_rows))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost reduction", f"{cost_reduction:.1f}%")
    c2.metric("Service level (achieved)", f"{achieved_service:.1f}%")
    c3.metric("Forecast error (MAPE)", f"{overall_mape:.1f}%")
    c4.metric("SKU x Warehouse scenarios", f"{len(results)}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline cost", x=results["sku"] + " / " + results["warehouse"],
                              y=results["baseline_cost"], marker_color="#EF553B"))
        fig.add_trace(go.Bar(name="Optimized cost", x=results["sku"] + " / " + results["warehouse"],
                              y=results["optimized_cost"], marker_color="#00CC96"))
        fig.update_layout(barmode="group", title="Total cost by SKU x Warehouse",
                           xaxis_tickangle=-45, height=420, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=results["sku"] + " / " + results["warehouse"],
                               y=results["cost_reduction_pct"], marker_color="#636EFA"))
        fig2.add_hline(y=cost_reduction, line_dash="dash", line_color="gray",
                        annotation_text=f"Portfolio avg: {cost_reduction:.1f}%")
        fig2.update_layout(title="Cost reduction % by SKU x Warehouse",
                            xaxis_tickangle=-45, height=420, yaxis_title="% reduction")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Full results table")
    display_df = results.copy()
    display_df["baseline_service_level"] = (display_df["baseline_service_level"] * 100).round(1)
    display_df["optimized_service_level"] = (display_df["optimized_service_level"] * 100).round(1)
    display_df["cost_reduction_pct"] = display_df["cost_reduction_pct"].round(1)
    display_df[["baseline_cost", "optimized_cost"]] = display_df[["baseline_cost", "optimized_cost"]].round(0)
    st.dataframe(
        display_df[["sku", "warehouse", "baseline_cost", "baseline_service_level",
                     "optimized_cost", "optimized_service_level", "cost_reduction_pct"]]
        .rename(columns={
            "baseline_cost": "Baseline cost", "baseline_service_level": "Baseline SL %",
            "optimized_cost": "Optimized cost", "optimized_service_level": "Optimized SL %",
            "cost_reduction_pct": "Cost reduction %",
        }),
        use_container_width=True, hide_index=True,
    )

# ============================================================= TAB 2 =====
with tab2:
    st.subheader("Explore a single SKU x Warehouse pair")
    col1, col2 = st.columns(2)
    sku = col1.selectbox("SKU", SKUS, key="pair_sku")
    wh = col2.selectbox("Warehouse", WAREHOUSES, key="pair_wh")

    g, train_days, test_days, fc, err = forecast_for_pair(df, sku, wh)

    st.markdown(f"**Forecast accuracy for {sku} @ {wh}: MAPE = {err:.1f}%**")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=g["date"], y=g["demand"], name="Actual demand",
                                 line=dict(color="#636EFA", width=1)))
    forecast_dates = g["date"].iloc[train_days:train_days + test_days]
    fig_fc.add_trace(go.Scatter(x=forecast_dates, y=fc, name="Holt-Winters forecast",
                                 line=dict(color="#EF553B", width=2, dash="dash")))
    fig_fc.add_vline(x=g["date"].iloc[train_days], line_dash="dot", line_color="gray",
                      annotation_text="train/test split")
    fig_fc.update_layout(title="Demand history & held-out forecast", height=380)
    st.plotly_chart(fig_fc, use_container_width=True)

    lt_mean, lt_std = g["lead_time_mean"].iloc[0], g["lead_time_std"].iloc[0]
    unit_cost = SKU_UNIT_COST[sku]
    train = g["demand"].iloc[:-EVAL_HORIZON].values
    d_mean, d_std = _series_stats(train)

    sl_for_pair = st.slider("Target service level for this pair", 0.50, 0.99,
                             DEFAULT_SERVICE_LEVEL, 0.01, key="pair_sl")
    params = compute_policy(
        d_mean=d_mean, d_std=d_std, lt_mean=lt_mean, lt_std=lt_std,
        annual_demand=d_mean * 365, unit_cost=unit_cost, ordering_cost=ORDERING_COST,
        holding_cost_rate=HOLDING_COST_RATE, service_level=sl_for_pair,
    )

    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    kcol1.metric("Reorder point", f"{params.reorder_point:,.0f} units")
    kcol2.metric("Safety stock", f"{params.safety_stock:,.0f} units")
    kcol3.metric("Order quantity (EOQ)", f"{params.order_quantity:,.0f} units")
    kcol4.metric("z-score", f"{params.z:.2f}")

    # single illustrative simulated inventory trajectory
    rng = np.random.default_rng(3)
    schedule = [{"start_day": 0, "end_day": EVAL_HORIZON,
                 "reorder_point": params.reorder_point, "order_qty": params.order_quantity}]
    eval_pool = g["demand"].iloc[-EVAL_HORIZON:].values

    on_hand = params.reorder_point + params.order_quantity * 0.5
    pending, trajectory = [], []
    for day in range(EVAL_HORIZON):
        arrivals = [q for a, q in pending if a == day]
        for q in arrivals:
            on_hand += q
        pending = [(a, q) for a, q in pending if a != day]
        d = rng.choice(eval_pool)
        on_hand = max(on_hand - d, 0)
        trajectory.append(on_hand)
        pos = on_hand + sum(q for _, q in pending)
        if pos <= params.reorder_point:
            lt = max(1, int(round(rng.normal(lt_mean, lt_std))))
            pending.append((day + lt, params.order_quantity))

    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(y=trajectory, name="On-hand inventory", line=dict(color="#00CC96")))
    fig_inv.add_hline(y=params.reorder_point, line_dash="dash", line_color="#EF553B",
                       annotation_text="Reorder point")
    fig_inv.add_hline(y=params.safety_stock, line_dash="dot", line_color="#AB63FA",
                       annotation_text="Safety stock")
    fig_inv.update_layout(title="Simulated inventory trajectory (one Monte Carlo replication)",
                           height=380, xaxis_title="Day", yaxis_title="Units on hand")
    st.plotly_chart(fig_inv, use_container_width=True)

# ============================================================= TAB 3 =====
with tab3:
    st.subheader("Methodology — how each number is produced")
    st.markdown(r"""
**1. Demand forecasting** — Holt-Winters exponential smoothing (additive
damped trend + weekly seasonality) fit per SKU x warehouse series.
Accuracy is measured as **MAPE** on a held-out window, on 7-day
rolling-averaged actual vs. forecast (standard practice for noisy daily
demand).

**2. Reorder point & safety stock** — classic formulas, accounting for
*both* demand and lead-time uncertainty:

- Safety stock: `SS = z * sqrt(LT_mean * sigma_d^2 + d_mean^2 * sigma_LT^2)`
- Reorder point: `ROP = d_mean * LT_mean + SS`
- Order quantity (EOQ): `Q = sqrt(2 * D * S / H)`

  where `z` is the standard-normal critical value for the target service
  level, `D` is annual demand, `S` is the fixed cost per order, and `H`
  is the annual holding cost per unit.

**3. Monte Carlo simulation** — for a given (reorder point, order
quantity) policy, demand is bootstrap-resampled from the actual
historical distribution (not assumed normal) and lead times are drawn
from a normal distribution around each warehouse's mean/variability.
150 independent replications are averaged to get stable estimates of
service level, average on-hand inventory, order frequency, and total
cost (holding + ordering + stockout penalty).

**4. Rolling-horizon planning** — instead of setting the policy once,
the reorder point / safety stock / order quantity are **re-computed
every {REPLAN_EVERY} days** from a trailing window of the latest
demand, and handed to a single continuous simulation (inventory and
in-transit orders carry across re-planning boundaries — the system is
never artificially "reset to full").

**5. Baseline** — represents how these SKUs are typically managed
*before* a formal optimization project: reorder at exactly the expected
lead-time demand with **no statistical safety-stock buffer**, and order
in a fixed lot size. This is a common, realistic starting point (sizing
stock to the "average case" and ignoring variability), which is why it
runs into stockouts and higher total cost relative to the optimized
policy.

**Why the achieved service level isn't exactly the target input:**
because demand is bootstrapped from the real (non-normal) distribution
and lead times are themselves stochastic, the textbook z-score formula
is an approximation — the dashboard reports the *actual measured*
service level from simulation, which is the honest number to quote.
""".replace("{REPLAN_EVERY}", str(REPLAN_EVERY)))
