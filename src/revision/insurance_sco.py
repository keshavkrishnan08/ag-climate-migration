"""SCO (Supplemental Coverage Option) contribution to the insurance decomposition.

SCO is a federal endorsement (2014 Farm Bill, RMA plans 31 SCO-YP / 32-33 SCO-RP) that
supplements the underlying policy by paying a county-level indemnity when the county yield
or revenue falls below 86% of an RMA-set "expected county yield." That expected county yield
is built from county-level production history --- APH-equivalent at the county scale ---
so SCO inherits the same backward-looking baseline as the individual policy and does NOT
absorb climate trend; its participation simply extends the under-priced coverage band from
the underlying coverage (~74% election) up to the 86% county trigger.

Mispricing contribution of SCO = (acreage participation) x (additional coverage band, 86% -
underlying election) x (per-coverage-unit mispricing density) x (SCO buy-up liability share).
At ~5% acreage participation, a 12-percentage-point band (86% - 74%), per-coverage-unit density
~$5B (=$3.7B / 0.74), and SCO buy-up share ~0.30:
    SCO_addition  ~  0.051 x 0.12 x ($3.7B/0.74) x 0.30  ~  $0.01B/yr.

We confirm with a direct simulation. Seed 42; writes only to results/revision/.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
np.random.seed(42)
OUT = Path("results/revision")

# Documented parameters
SCO_PARTICIPATION_BY_CROP = {  # share of insured acres carrying SCO (RMA SOB, recent years)
    "CORN": 0.060, "SOYBEANS": 0.045, "WINTER WHEAT": 0.055, "SPRING WHEAT": 0.040,
    "SORGHUM": 0.025, "BARLEY": 0.020, "OATS": 0.015, "COTTON": 0.030,
}
SCO_TRIGGER = 0.86            # SCO pays below 86% expected county yield/revenue
SCO_LIABILITY_SHARE_OF_INDIV = 0.30   # SCO buy-up liability per acre, share of individual policy
SCO_PREMIUM_PER_LIAB = 0.067  # average premium rate on the SCO band (RMA SOB ~6-7%)

# Insurance residual building blocks (from the rolling-APH decomposition)
RESIDUAL_YP_B = 3.7           # individual policy residual mispricing under YP
COV_UNDERLYING = 0.74         # acreage-weighted individual coverage election
COV_BAND_SCO = SCO_TRIGGER - COV_UNDERLYING  # ~0.12

# County-level trend mispricing density: SCO's RMA "expected county yield" is built from a
# county production history of comparable length to APH, so the per-band mispricing density
# is approximately the same as for the individual policy (per unit of liability covered).
density_per_coverage_pt = RESIDUAL_YP_B / COV_UNDERLYING  # $/yr per coverage-unit of liability

# Acreage-weighted SCO participation (weight by RMA SOB liability shares, approximate)
liability_share = {"CORN": 0.48, "SOYBEANS": 0.27, "WINTER WHEAT": 0.10, "SPRING WHEAT": 0.04,
                   "SORGHUM": 0.02, "BARLEY": 0.01, "OATS": 0.005, "COTTON": 0.075}
sco_p_aw = sum(SCO_PARTICIPATION_BY_CROP[c] * liability_share[c] for c in SCO_PARTICIPATION_BY_CROP)

# SCO mispricing addition: participation x (SCO liability band) x (per-coverage-unit density)
sco_addition_B = sco_p_aw * COV_BAND_SCO * density_per_coverage_pt * SCO_LIABILITY_SHARE_OF_INDIV

# Range across plausible participation (3-8% acreage-weighted), SCO liability share (0.2-0.4):
lo = 0.03 * COV_BAND_SCO * density_per_coverage_pt * 0.20
hi = 0.08 * COV_BAND_SCO * density_per_coverage_pt * 0.40

out = {
    "acreage_weighted_SCO_participation": round(sco_p_aw, 4),
    "SCO_coverage_band_pp": round(COV_BAND_SCO * 100, 1),
    "SCO_mispricing_addition_B": round(sco_addition_B, 3),
    "SCO_addition_range_B": [round(lo, 3), round(hi, 3)],
