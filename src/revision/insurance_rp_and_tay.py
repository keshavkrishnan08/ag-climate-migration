"""R2-2d: full Revenue-Protection put with harvest-price reset (vs the yield-only
put), and R2-3b: residual mispricing across the full TAY-participation range.

R2-2d: RP guarantees revenue at APH*cov*max(P_proj, P_harvest); indemnity =
max(guarantee - yield*P_harvest, 0), a price-yield interaction. We Monte-Carlo the
harvest price (lognormal, negatively correlated with yield -- the natural hedge)
and recompute the CLIMATE mispricing (forward vs rolling-APH yield) under RP, then
compare to the yield-only (YP) put. If the RP price channel is climate-neutral, the
two climate-mispricing aggregates coincide, validating the closed-form yield put.

R2-3b: TAY participation is an endorsement not flagged in the SOB-coverage file, so
rather than assume a single rate we report the reform-eliminable residual across the
entire participation range (0-100%), bounding the headline without an assumption.

Seed 42; writes only to results/revision/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from insurance_rolling_aph import build_rma_county_crop, PRICE, TAY_LAG_YEARS
from insurance_fast import build_paths, simulate_fast, TAY_PARTICIPATION
OUT = ROOT / "results" / "revision"
np.random.seed(42)
WIN = (2040, 2050)
PRICE_VOL = 0.20      # harvest-price lognormal vol (RMA volatility factors ~0.15-0.30)
YIELD_PRICE_CORR = -0.30   # natural hedge: low yields <-> high prices


def rp_vs_yp_climate_mispricing():
    """Aggregate climate mispricing under YP (yield-only put) vs RP (revenue put
    with harvest-price reset), to test whether the RP price channel is climate-neutral.
    Monte Carlo per county-crop over the headline window."""
    rma = build_rma_county_crop()
    paths, cv = build_paths("SSP245")
    keys = rma[["fips", "crop"]].drop_duplicates()
    paths = paths.merge(keys, on=["fips", "crop"], how="inner")
    wide = paths.pivot_table(index=["fips", "crop"], columns="year", values="y", aggfunc="first").sort_index(axis=1)
    meta = rma.set_index(["fips", "crop"]).reindex(wide.index)
    cvv = cv.set_index(["fips", "crop"]).reindex(wide.index)["cv"].fillna(0.20).values
    obs = [c for c in wide.columns if c <= 2024]
    aph_frozen = wide[obs].mean(axis=1).values
    crops = pd.Index(wide.index.get_level_values("crop"))
    price = crops.map(lambda c: PRICE.get(c, 4.0)).to_numpy(float)
    cov = meta["cov_wt"].values; prem = meta["prem_per_acre"].values; acres = meta["insured_acres"].values
    valid = np.isfinite(aph_frozen) & (aph_frozen > 0) & np.isfinite(cov) & np.isfinite(prem)

    N = 4000
    rng = np.random.default_rng(42)
    yp_flow, rp_flow = [], []
    for T in range(WIN[0], WIN[1] + 1):
        wcols = [y for y in range(T - 10, T) if y in wide.columns]
        roll = np.nanmean(wide[wcols].values, axis=1)
        true_y = wide[T].values if T in wide.columns else roll
        # standardized yield draws shared across YP/RP for variance reduction
        z = rng.standard_normal((len(aph_frozen), N))
        zp = rng.standard_normal((len(aph_frozen), N))
        for label, aph in [("true", true_y), ("roll", roll)]:
            sd = aph_frozen * cvv                      # yield sd (bu)
            ydraw = np.maximum(aph[:, None] + sd[:, None] * z, 0) if label == "roll" \
                    else np.maximum(true_y[:, None] + sd[:, None] * z, 0)
            # YP: indemnity = price * max(K_yield - yield, 0), K_yield = APH_roll*cov
            K_y = roll * cov
            ind_yp = price[:, None] * np.maximum(K_y[:, None] - ydraw, 0)
            # RP: harvest price lognormal, corr with yield; guarantee revenue with reset
            eps = YIELD_PRICE_CORR * z + np.sqrt(1 - YIELD_PRICE_CORR**2) * zp
            ph = price[:, None] * np.exp(PRICE_VOL * eps - 0.5 * PRICE_VOL**2)
            guar = roll[:, None] * cov[:, None] * np.maximum(price[:, None], ph)   # harvest-price reset
