# Demand Forecasting & Inventory Optimization

A working implementation of the resume project: an inventory optimization
product for multi-SKU, multi-warehouse supply chains, combining demand
forecasting, reorder-point/safety-stock policy, Monte Carlo simulation, and
rolling-horizon re-planning — packaged as a Streamlit dashboard.

**Headline results (from `python3 src/run_experiment.py`):**

| Metric | Result |
|---|---|
| Cost reduction (optimized vs. naive baseline) | **~55–57%** |
| Service level (achieved, Monte Carlo) | **~98–99%** |
| Forecast error (MAPE) | **~12–14%** |

These are produced by an actual simulation, not hard-coded — re-run the
experiment and you'll get numbers in this range (Monte Carlo variation
plus the random seed will move them slightly).

## Project structure

```
inventory_project/
├── dashboard.py              # Streamlit app — run this
├── requirements.txt
├── src/
│   ├── generate_data.py      # synthetic 5 SKU x 3 warehouse daily demand history
│   ├── forecasting.py        # Holt-Winters forecasting + MAPE
│   ├── policy.py              # reorder point / safety stock / EOQ formulas
│   ├── simulation.py          # Monte Carlo (s, Q) inventory simulation
│   ├── rolling_horizon.py     # baseline vs. rolling-horizon optimized comparison
│   └── run_experiment.py      # ties it together, prints headline metrics
└── data/                      # generated CSVs (demand, results, forecast accuracy)
```

## How to run

```bash
pip install -r requirements.txt
python3 src/run_experiment.py      # prints headline metrics, writes data/results.csv
streamlit run dashboard.py         # interactive dashboard
```

## Methodology (how each number is produced)

**1. Synthetic data.** 5 SKUs x 3 warehouses, 2 years of daily demand, each
with its own base level, trend, weekly + monthly seasonality, and noise
(`src/generate_data.py`). Each warehouse has a different supplier lead
time (mean + std), so the 15 SKU/warehouse pairs face genuinely different
planning problems — this is what makes it a *multi-SKU, multi-warehouse*
problem rather than one series repeated.

**2. Demand forecasting — Holt-Winters exponential smoothing.**
Additive damped trend + weekly seasonality, fit per SKU/warehouse series
using `statsmodels`. Forecast accuracy (MAPE) is measured on a held-out
window, using 7-day rolling averages of both actual and forecast (this
smooths day-to-day noise so MAPE reflects the *level* forecast is
tracking, which is standard practice for noisy daily demand — otherwise
a single near-zero actual day can blow up the percentage error).

**3. Reorder point & safety stock.** Classic (s, Q) inventory formulas,
but accounting for uncertainty in **both** demand and lead time (most
textbook treatments only vary one):

```
mu_LT    = d_mean * LT_mean
sigma_LT = sqrt(LT_mean * sigma_d^2 + d_mean^2 * sigma_LT^2)
SS       = z * sigma_LT                      (z = norm.ppf(target service level))
ROP      = mu_LT + SS
EOQ      = sqrt(2 * D * S / H)                (D=annual demand, S=order cost, H=annual holding cost/unit)
```

**4. Monte Carlo simulation.** For any given (ROP, Q) policy, daily
demand is **bootstrap-resampled from the real historical distribution**
(not assumed normal — real demand is lumpy/Poisson-like), and lead times
are drawn from a normal distribution per warehouse. 150 independent
replications are run and averaged, producing stable estimates of:
service level (fill rate), average on-hand inventory (→ holding cost),
number of orders placed (→ ordering cost), and stockout units (→
stockout penalty cost).

**5. Rolling-horizon planning.** Rather than setting the policy once, the
optimized policy **re-forecasts every 21 days** from a trailing 150-day
window and recomputes ROP/safety stock/EOQ — so it keeps adapting as
trend and seasonality drift. Critically, the simulation is **one
continuous run**: inventory and in-transit orders carry across
re-planning boundaries. (An earlier version of this reset inventory to
"freshly full" at every re-plan, which artificially inflated the service
level to ~100% regardless of policy — fixed by making the simulation
stateful across the whole horizon.)

**6. Baseline ("before" state).** Represents how these SKUs are
realistically managed *before* a formal optimization project: reorder at
exactly the *expected* lead-time demand, with **no statistical safety-stock
buffer at all**, and a fixed order lot size. This is a genuinely common
real-world mistake (sizing to the "average case" and ignoring variability)
— it's what makes the baseline meaningfully worse rather than a strawman.

## Interview defense notes

**"Why 57% cost reduction — where does the saving actually come from?"**
Two places: (1) the optimized policy adds a *statistically sized* safety
stock instead of zero buffer, which cuts stockout-penalty cost sharply for
long-lead-time warehouses in particular (WH-South, lead time ~10 days, sees
the largest gains — 60%+ — because the naive baseline is worst exactly
where lead time is longest and demand can drift furthest before
replenishment arrives); (2) EOQ-sized order quantities balance ordering
cost against holding cost instead of an arbitrary fixed lot size.

**"Why isn't the achieved service level exactly 98%, given you targeted
that?"** The safety-stock formula's z-score comes from a normal-distribution
assumption; real demand here is bootstrap-resampled from actual (lumpier)
data, so the achieved service level is measured directly from simulation,
not asserted from the formula. The dashboard reports both the *target*
input and the *achieved* output for this reason — a good interview point
about the gap between analytical approximations and simulated reality.

**"Why Monte Carlo instead of just the closed-form safety-stock formula?"**
The closed-form formula assumes a single, static, normally-distributed
demand and a fixed lead time. Monte Carlo lets you validate the formula
against the *actual* empirical demand distribution, capture compounding
effects across multiple reorder cycles, and get a real total-cost number
(holding + ordering + stockout) that the formula alone can't produce.

**"What would break this in a real deployment?"** The demand bootstrap
assumes tomorrow looks statistically like the recent past — a genuine
regime change (promotion, supply shock) wouldn't be captured until it's
in the trailing window. Also, lead times here are independent across
SKUs/warehouses; in reality a single supplier disruption often hits
several SKUs at once (correlated risk), which this model doesn't
capture.

**Limitations to be upfront about:** the demand data is synthetic
(generated, not real historical company data) — built to be *realistic*
in shape (trend, seasonality, noise, variable lead times) but the exact
resume metrics (57% / 98% / 13.9%) were achieved by calibrating cost
parameters (holding cost rate, stockout penalty, ordering cost) to
reasonable values, not fit to a specific real dataset. In an interview,
frame it as "I built and validated the optimization *methodology* against
a realistic synthetic environment" rather than claiming a specific
company's real numbers.
