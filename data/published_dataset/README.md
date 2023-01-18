# County-Level Agricultural Climate Migration Dataset

## Description

County-level projections of climate-driven agricultural reorganization in the United States, 2025–2050. Accompanies:

> Krishnan, K. (2026). The economic cost of agricultural climate migration in the United States. *Nature Climate Change*.

Six CSV files cover yield projections, climate projections, stranded asset valuations, rural decline indicators, insurance mispricing, and northern opportunity scores for all US agricultural counties.

---

## Files

### 1. `county_yield_projections.csv`
Projected yields for 8 crops × 2,902 counties × 26 years × 2 scenarios (SSP2-4.5, SSP3-7.0). 10-GCM ensemble median with 10th/90th percentile bounds.

| Column | Description |
|--------|-------------|
| `fips` | 5-digit county FIPS code |
| `county_name` | County name (Census Gazetteer 2023) |
| `state` | State postal abbreviation |
| `year` | Projection year (2025–2050) |
| `crop` | Crop type (corn, soybeans, wheat, cotton, sorghum, barley, oats, hay) |
| `scenario` | Climate scenario (SSP245 or SSP370) |
| `yield_projected_bu_acre` | Projected yield including climate and technology trend (bu/acre) |
| `yield_baseline_bu_acre` | Baseline yield (technology trend, no climate change) (bu/acre) |
| `climate_impact_bu_acre` | Climate-only impact = projected − baseline (bu/acre) |
| `yield_p10` | 10th percentile across GCM ensemble (bu/acre) |
| `yield_p90` | 90th percentile across GCM ensemble (bu/acre) |
| `acres_harvested` | Harvested acreage (acres, NASS 2018–2022 average) |

---

### 2. `county_climate_projections.csv`
Downscaled climate projections at county resolution. Delta-method from 10 CMIP6 GCMs (SSP2-4.5) and 9 GCMs (SSP3-7.0).

| Column | Description |
|--------|-------------|
| `fips` | 5-digit county FIPS code |
