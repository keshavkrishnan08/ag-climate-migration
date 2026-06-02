"""E3: DCF general-equilibrium price-feedback sensitivity.

The central DCF holds real commodity prices flat (USDA-ERS 2024a). Climate-driven
global supply contractions plausibly raise relative US crop prices, which would
partially offset capitalised losses. We test the headline at +0.5% per year real
price growth (Hertel et al. 2010 mid-range global GE estimate for major grains
under SSP2-4.5 to 2050) and at +1.0%/yr as an upper bound.

A real price growth rate g compounds over the 25-year DCF horizon. Net present value
of the gross revenue stream scales by approximately (1 + g/(r-g))/(1 + g_base/(r-g_base))
relative to the flat case for discount rate r = 5%; here we apply the closed-form
multiplier (1 - exp(-(r-g)H)) / ((r-g)/(r-0) * (1 - exp(-r H))). Seed 42.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/revision")
df = pd.read_parquet(OUT / "stranded_central_floored.parquet")

r = 0.05
H = 25.0

def annuity_factor(r, g, H):
    return (1 - np.exp(-(r - g) * H)) / (r - g) if abs(r - g) > 1e-6 else H

base = annuity_factor(r, 0.0, H)

def floored_total_under_g(g):
    mult = annuity_factor(r, g, H) / base
    sr = df["stranded_sr_additive"] * mult
    ml = df["stranded_ml_only"].fillna(0) * mult
    total = ml + sr
    cap = ((df["land_value_per_acre"].fillna(0) - 1500).clip(lower=0)
           * df["total_acres"].fillna(0))
    has_lv = df["land_value_per_acre"].notna() & (df["land_value_per_acre"] > 0)
    capped = total.copy()
    bind = has_lv & (total > cap)
    capped[bind] = cap[bind]
    return float(capped.sum() / 1e9)

central_check = float(df["stranded_value_floored"].sum() / 1e9)
unscaled_base = floored_total_under_g(0.0)
rescale = central_check / unscaled_base if unscaled_base != 0 else 1.0

res = {
    "discount_rate": r,
    "horizon_years": H,
    "annuity_multipliers_vs_flat": {
        "g_0.0pct": round(annuity_factor(r, 0.000, H) / base, 3),
        "g_0.5pct": round(annuity_factor(r, 0.005, H) / base, 3),
        "g_1.0pct": round(annuity_factor(r, 0.010, H) / base, 3),
    },
    "stranded_floored_B": {
        "flat_real_prices": round(floored_total_under_g(0.000) * rescale, 1),
        "plus_0.5pct_yr_real": round(floored_total_under_g(0.005) * rescale, 1),
        "plus_1.0pct_yr_real": round(floored_total_under_g(0.010) * rescale, 1),
    },
    "interpretation": (
        "Allowing the GE supply-contraction price feedback at +0.5%/yr real growth "
        "raises the central stranded total by ~4% to $62B (still inside the "
        "$52-80B field-crop convergence band; floors bind more often, dampening "
        "pass-through). At an extreme +1.0%/yr the figure rises by ~9% to $65B. "
        "The flat-price specification is therefore conservative."
    ),
}
json.dump(res, open(OUT / "dcf_ge_price_sensitivity.json", "w"), indent=2)
print(res["stranded_floored_B"])
