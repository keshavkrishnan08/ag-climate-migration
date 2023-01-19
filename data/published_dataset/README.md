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
| `county_name` | County name |
| `state` | State postal abbreviation |
| `year` | Projection year |
| `scenario` | Climate scenario (SSP245 or SSP370) |
| `tmax_july_projected_F` | Projected July maximum temperature (°F) |
| `delta_tmax_july_F` | Change in July Tmax vs 1990–2020 baseline (°F) |
| `precip_growing_projected_mm` | Projected growing-season precipitation (mm/month) |
| `delta_precip_mm` | Change in growing-season precipitation vs baseline (mm/month) |
| `tmax_july_p10` | 10th percentile July Tmax across GCM ensemble (°F) |
| `tmax_july_p90` | 90th percentile July Tmax across GCM ensemble (°F) |
| `n_gcms` | Number of GCMs contributing to ensemble |

---

### 3. `county_stranded_assets.csv`
Stranded farmland value per county under three independent valuation methods. One row per county.

| Column | Description |
|--------|-------------|
| `fips` | 5-digit county FIPS code |
| `county_name` | County name |
| `state` | State postal abbreviation |
| `stranded_dcf_conservative_usd` | Stranded value, DCF method, SSP2-4.5, 4% discount (2023 USD) |
| `stranded_dcf_central_usd` | Stranded value, DCF method, SSP3-7.0, 4% discount (2023 USD) |
| `stranded_hedonic_usd` | Stranded value, hedonic method, 2050 horizon, SSP2-4.5 (2023 USD) |
| `stranded_per_acre_usd` | Stranded value per acre, DCF conservative (2023 USD/acre) |
| `stranded_fraction` | Stranded value as fraction of total land value |
| `land_value_per_acre_usd` | Current farmland value per acre (2023 USD, NASS) |
| `total_farm_acres` | Total harvested farmland acres |

