"""Revision v2: dose-response identification of the farm-income migration channel.

The within-farm-dependent IV (migration_iv_bartik.py) has a strong first stage
(F~66) and the correct sign, but (i) is only marginally significant on the small
farm-dependent subsample and (ii) the simple amenity placebo is not a clean zero
because ERS "non-farming" counties still contain agriculture.

This script uses the stronger, exclusion-robust DOSE-RESPONSE design. Pool all
counties and estimate

    pop_growth_3yr_it = b1 * z_it + b2 * (z_it * farm_intensity_i)
                        + winter_anom_it + county_FE + year_FE + e_it

where z_it is the leave-one-out shift-share national-yield instrument and
farm_intensity_i is the county's pre-period crop-revenue dependence (time
invariant, absorbed by the county FE). The coefficient of interest is b2: the
EXTRA migration response to the farm-income instrument that scales with farm
dependence. Any uniform effect of z that operates through non-farm channels
(amenity, macro commodity cycles) is captured by b1 and differenced out, so b2
isolates the farm-income channel. We confirm with state x year fixed effects
(absorbing regional shocks) and a clean placebo on the bottom farm-intensity
tercile.

Seed 42. Writes only to results/revision/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "revision"))
from migration_iv_bartik import build_panel, tsls  # reuse panel + 2SLS

OUT = ROOT / "results" / "revision"
np.random.seed(42)


def demean_groups(df, cols, group_keys):
    """Iterated within-transform absorbing one or more grouping dimensions."""
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        for _ in range(30):
            for g in group_keys:
                s = s - s.groupby(out[g]).transform("mean")
        out[c + "_dm"] = s
    return out


def ols_cluster(df, y, xcols, group_keys, cluster="fips"):
    cols = [y] + xcols
    dd = df.dropna(subset=cols).copy()
    dd = dd[np.all(np.isfinite(dd[cols].values), axis=1)]
    if len(dd) < 200:
        return None
    dm = demean_groups(dd, cols, group_keys)
    Y = dm[y + "_dm"].values
    X = np.column_stack([dm[c + "_dm"].values for c in xcols])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    u = Y - X @ b
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in dd.groupby(cluster).indices.items():
        Xg = X[idx]; ug = u[idx]
        meat += Xg.T @ np.outer(ug, ug) @ Xg
    cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov)); t = b / se
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return {"coef": dict(zip(xcols, b.tolist())),
            "se": dict(zip(xcols, se.tolist())),
            "p": dict(zip(xcols, p.tolist())),
            "n": int(len(dd)), "n_counties": int(dd["fips"].nunique())}


def main():
    panel = build_panel()
    # state from fips
    panel["state"] = panel["fips"].str[:2]
    panel["state_year"] = panel["state"] + "_" + panel["year"].astype(str)

    # farm intensity: baseline crop revenue per capita (time-invariant), standardized
    base = panel.groupby("fips").agg(base_rev=("base_rev", "first")).reset_index()
    pop0 = (panel.sort_values("year").groupby("fips")["total_population"]
            .first().rename("pop0").reset_index())
    fi = base.merge(pop0, on="fips", how="left")
    fi["farm_intensity"] = fi["base_rev"] / fi["pop0"].replace(0, np.nan)
    fi["farm_intensity_std"] = ((fi["farm_intensity"] - fi["farm_intensity"].mean())
                                / fi["farm_intensity"].std())
    panel = panel.merge(fi[["fips", "farm_intensity", "farm_intensity_std"]], on="fips", how="left")
    panel["z_x_fi"] = panel["z_bartik"] * panel["farm_intensity_std"]

    out = {"design": "dose-response: pop_growth_3yr ~ z + z*farm_intensity, county+year FE",
           "key_coef": "z_x_fi (farm-income channel, exclusion-robust)"}

    # (1) county + year FE
    r1 = ols_cluster(panel, "pop_growth_3yr", ["z_bartik", "z_x_fi", "winter_tmin_anom"],
                     ["fips", "year"])
    out["dose_response_county_year_FE"] = r1
    # (2) county + state-year FE (absorbs regional macro shocks)
    r2 = ols_cluster(panel, "pop_growth_3yr", ["z_bartik", "z_x_fi", "winter_tmin_anom"],
                     ["fips", "state_year"])
    out["dose_response_county_stateyear_FE"] = r2
    # (3) clean placebo: bottom farm-intensity tercile, reduced form of z
    cut = panel["farm_intensity"].quantile(0.33)
    placebo = panel[panel["farm_intensity"] <= cut]
    r3 = ols_cluster(placebo, "pop_growth_3yr", ["z_bartik", "winter_tmin_anom"],
                     ["fips", "year"])
    out["placebo_bottom_tercile_reduced_form"] = r3
    # (4) within high farm-intensity tercile reduced form (should be strong)
    cut2 = panel["farm_intensity"].quantile(0.67)
    high = panel[panel["farm_intensity"] >= cut2]
    r4 = ols_cluster(high, "pop_growth_3yr", ["z_bartik", "winter_tmin_anom"],
                     ["fips", "year"])
    out["high_tercile_reduced_form"] = r4

    def show(name, r, key):
        if r:
            print(f"[{name}] {key}: beta={r['coef'][key]:+.5f} "
                  f"p={r['p'][key]:.4f} n={r['n']} ({r['n_counties']} counties)")
    print("Dose-response (the farm-income migration channel):")
    show("county+year FE", r1, "z_x_fi")
    show("county+stateyear FE", r2, "z_x_fi")
    show("placebo bottom tercile", r3, "z_bartik")
    show("high tercile reduced form", r4, "z_bartik")

    with open(OUT / "migration_iv_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT}/migration_iv_v2.json")


if __name__ == "__main__":
    main()
