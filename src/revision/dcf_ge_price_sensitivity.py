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
