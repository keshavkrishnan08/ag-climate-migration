"""E52: Spatial-block (regionally correlated) yield-impact perturbation.

Reviewer concern: a uniform +-25% shift in projected yield impact does not
address spatially correlated model bias. A 25% miss concentrated in the
Mississippi Delta is not interchangeable with a 25% miss spread uniformly.
We apply regional 25% perturbations to the Schlenker-Roberts additive
penalty by USDA Farm Production Region (FPR) and recompute the central
alternate-use-floored DCF.

Regions tested (one at a time, +25% then -25%):
  Delta, Southern Plains, Corn Belt, Northern Plains, Lake States,
  Appalachia, Southeast, Mountain, Pacific, Northeast.

The headline dollar range survives if no single regional block moves the
central below the $52--80B convergence band by more than a few B. Seed 42.
"""
import json
from pathlib import Path
import pandas as pd

OUT = Path("results/revision")
df = pd.read_parquet(OUT / "stranded_central_floored.parquet").copy()


def state_from_fips(fips):
    return str(fips)[:2]


# USDA Farm Production Regions, state FIPS -> region.
FPR = {
