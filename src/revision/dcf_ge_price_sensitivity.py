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
