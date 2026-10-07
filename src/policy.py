"""
policy.py
---------
Classic inventory-control formulas:

  Safety Stock (SS)   = z * sigma_LT
  Reorder Point (ROP) = mu_LT + SS
  EOQ                 = sqrt(2 * D * S / H)

where, for demand with mean `d_mean`/day and std `d_std`/day, and a lead
time with mean `lt_mean` days and std `lt_std` days, the demand-during-lead-
time mean and variance combine both demand uncertainty AND lead-time
uncertainty (the standard formula used when both are stochastic):

  mu_LT    = d_mean * lt_mean
  sigma_LT = sqrt(lt_mean * d_std**2 + d_mean**2 * lt_std**2)

`z` is the standard-normal critical value for the target service level
(e.g. z ~= 2.054 for a 98% cycle service level).
"""

from dataclasses import dataclass
from scipy.stats import norm
import numpy as np


@dataclass
class PolicyParams:
    reorder_point: float
    safety_stock: float
    order_quantity: float
    z: float


def z_for_service_level(service_level: float) -> float:
    return float(norm.ppf(service_level))


def compute_policy(
    d_mean: float, d_std: float, lt_mean: float, lt_std: float,
    annual_demand: float, unit_cost: float, ordering_cost: float,
    holding_cost_rate: float, service_level: float,
) -> PolicyParams:
    z = z_for_service_level(service_level)

    mu_lt = d_mean * lt_mean
    sigma_lt = np.sqrt(max(lt_mean, 0) * d_std ** 2 + d_mean ** 2 * lt_std ** 2)

    safety_stock = max(z * sigma_lt, 0)
    reorder_point = mu_lt + safety_stock

    holding_cost_per_unit = unit_cost * holding_cost_rate
    eoq = np.sqrt(max(2 * annual_demand * ordering_cost / max(holding_cost_per_unit, 1e-6), 0))
    eoq = max(eoq, 1.0)

    return PolicyParams(reorder_point=reorder_point, safety_stock=safety_stock,
                         order_quantity=eoq, z=z)
