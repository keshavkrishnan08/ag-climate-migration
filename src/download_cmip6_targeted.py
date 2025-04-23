"""Targeted CMIP6 download — only what the paper needs.

Downloads 5 diverse GCMs x ssp245 x 3 vars x milestone years.
Then linearly interpolates annual values for the projection pipeline.

Usage:
    python src/download_cmip6_targeted.py
"""

import sys
from pathlib import Path

# Reuse the main download machinery
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_cmip6 import process_one_file, CMIP6_DIR, VARIABLES, MODEL_VARIANTS
from loguru import logger
import time
import pandas as pd
import numpy as np
import os

# 5 GCMs spanning warm/cool/wet/dry for ensemble spread.
# All verified to have tasmax/tasmin/pr on the NASA NEX-GDDP-CMIP6 S3 bucket.
# CESM2 excluded: only has tas (mean temp), not tasmax/tasmin.
PRIORITY_MODELS = [
    'ACCESS-CM2',       # warm/wet (Australia)       — r1i1p1f1, gn
    'GFDL-ESM4',        # moderate (NOAA)            — r1i1p1f1, gr1
    'MIROC6',           # warm (Japan)               — r1i1p1f1, gn
    'MPI-ESM1-2-HR',    # moderate/cool (Germany)    — r1i1p1f1, gn
    'NorESM2-MM',       # moderate (Norway)          — r1i1p1f1, gn
]

# Milestone years — enough for interpolation + key decadal snapshots
MILESTONE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def download_targeted():
    """Download only the priority models x milestone years.

    Uses model-specific variant labels and grid labels from MODEL_VARIANTS
    to construct the correct S3 URLs for each GCM.
    """
    total = len(PRIORITY_MODELS) * len(VARIABLES) * len(MILESTONE_YEARS)
    logger.info(f"Targeted CMIP6: {len(PRIORITY_MODELS)} models x {len(VARIABLES)} vars "
                f"x {len(MILESTONE_YEARS)} years = {total} files")

    # Log the variant/grid info for each model
    for model in PRIORITY_MODELS:
        info = MODEL_VARIANTS.get(model, {})
        logger.info(f"  {model}: variant={info.get('variant', '???')}, "
                    f"grid={info.get('grid', '???')}")

    processed = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for model in PRIORITY_MODELS:
        model_info = MODEL_VARIANTS.get(model, {})
        variant = model_info.get('variant', 'r1i1p1f1')
        grid = model_info.get('grid', 'gn')

        for variable in VARIABLES:
            for year in MILESTONE_YEARS:
                output_path = CMIP6_DIR / f"{model}_ssp245_{variable}_{year}_conus_monthly.parquet"
                if output_path.exists():
                    skipped += 1
                    continue

                result = process_one_file(model, 'ssp245', variable, year,
                                          variant=variant, grid=grid)
                if not result.empty:
                    processed += 1
                else:
                    failed += 1

                done = processed + skipped + failed
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                logger.info(f"  [{done}/{total}] {model}/{variable}/{year} "
                            f"({processed} new, {skipped} cached, {failed} failed) "
                            f"ETA: {remaining/60:.0f} min")

    elapsed = time.time() - t0
    logger.info(f"\nTargeted download complete in {elapsed/60:.0f} min")
    logger.info(f"  Processed: {processed} | Cached: {skipped} | Failed: {failed}")

    # Interpolate annual values from milestones
    interpolate_annual()


def interpolate_annual():
    """Create annual interpolated files from milestone years.

    For each model x variable, linearly interpolates between milestone years
    to produce annual parquets needed by 05_project.py.
    """
    logger.info("\nInterpolating annual values from milestones...")

    for model in PRIORITY_MODELS:
        for variable in VARIABLES:
            milestones = {}
            for year in MILESTONE_YEARS:
                path = CMIP6_DIR / f"{model}_ssp245_{variable}_{year}_conus_monthly.parquet"
                if path.exists():
                    milestones[year] = pd.read_parquet(path)

            if len(milestones) < 2:
                logger.warning(f"  {model}/{variable}: <2 milestones, skipping interpolation")
                continue

            sorted_years = sorted(milestones.keys())

            for i in range(len(sorted_years) - 1):
                y_start = sorted_years[i]
                y_end = sorted_years[i + 1]
                df_start = milestones[y_start]
                df_end = milestones[y_end]

                for y in range(y_start + 1, y_end):
                    out_path = CMIP6_DIR / f"{model}_ssp245_{variable}_{y}_conus_monthly.parquet"
                    if out_path.exists():
                        continue

                    # Linear interpolation weight
                    w = (y - y_start) / (y_end - y_start)

                    # Interpolate the value column
                    df_interp = df_start.copy()
                    if 'value' in df_interp.columns and 'value' in df_end.columns:
                        df_interp['value'] = df_start['value'] * (1 - w) + df_end['value'] * w
                    df_interp['year'] = y

                    df_interp.to_parquet(out_path, index=False)

                logger.debug(f"  {model}/{variable}: interpolated {y_start+1}-{y_end-1}")

    # Count total files
    all_files = list(CMIP6_DIR.glob("*_ssp245_*_conus_monthly.parquet"))
    logger.info(f"Total CMIP6 parquets: {len(all_files)}")


if __name__ == '__main__':
    download_targeted()
