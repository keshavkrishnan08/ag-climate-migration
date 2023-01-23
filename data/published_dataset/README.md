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

---

### 4. `county_decline_indicators.csv`
Six rural decline indicators tracked for 1,824 agricultural counties, 2005–2023. Plus tipping-year estimates from two independent migration elasticity methods.

| Column | Description |
|--------|-------------|
| `fips` | 5-digit county FIPS code |
| `county_name` | County name |
| `state` | State postal abbreviation |
| `n_decline_indicators` | Count of active decline signals (0–6) |
| `yield_decline` | Statistically significant negative yield trend (0/1) |
| `pop_decline` | Statistically significant population decline (0/1) |
| `income_decline` | Statistically significant income decline (0/1) |
| `outmigration` | Above-median outmigration rate (0/1) |
| `school_decline` | Statistically significant school enrollment decline (0/1) |
| `hospital_closure` | County experienced rural hospital closure 2010–2023 (0/1) |
| `tipping_year_own_iv` | Projected year of economic tipping point, own IV method (SSP2-4.5) |
| `tipping_year_feng` | Projected year of economic tipping point, Feng et al. (2010) method (SSP2-4.5) |

---

### 5. `county_insurance_mispricing.csv`
Per-county insurance mispricing under forward-looking vs backward-looking (APH) actuarial frameworks. One row per county-crop.

| Column | Description |
|--------|-------------|
| `fips` | 5-digit county FIPS code |
| `county_name` | County name |
| `state` | State postal abbreviation |
| `crop` | Crop type |
| `mispricing_per_acre_usd` | Mispricing per insured acre (2023 USD/acre; negative = over-priced) |
| `direction` | "underpriced" or "overpriced" relative to climate-forward actuarial value |
| `insured_acres` | Acres under Federal crop insurance (RMA Summary of Business) |
| `annual_flow_usd` | Annual cross-subsidy flow = mispricing × insured acres (2023 USD/yr) |

---

### 6. `county_opportunity_frontier.csv`
Northern counties with projected agricultural opportunity and infrastructure capacity constraints (SSP2-4.5, 514 counties).

| Column | Description |
|--------|-------------|
| `fips` | 5-digit county FIPS code |
| `county_name` | County name |
| `state` | State postal abbreviation |
| `annual_opportunity_usd` | Total annual income opportunity (2023 USD/yr) |
| `yield_gain_usd` | Income gain from climate-driven yield improvements (2023 USD/yr) |
| `expansion_usd` | Income gain from expanded acreage (2023 USD/yr) |
| `infrastructure_gap_usd` | Estimated infrastructure investment needed to capture opportunity (2023 USD) |
| `infrastructure_capacity_ratio` | Current infrastructure capacity as fraction of projected production demand |

---

## Citation

```
Krishnan, K. (2026). The economic cost of agricultural climate migration
in the United States. Nature Climate Change.
```

---

## License

[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Units

- All monetary values in 2023 USD (deflated via BLS CPI-U, CPI₂₀₂₃ = 304.7)
- Temperatures in °F in climate projection files
- Temperatures in °C internally in yield model features (not in these CSVs)
- Yields in bushels per acre (bu/acre)
- Precipitation in mm/month

---

## Data Sources
