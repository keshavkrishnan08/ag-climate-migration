# Datasheet for County-Level Agricultural Climate Migration Dataset

Following Gebru et al. (2021) "Datasheets for Datasets"

## Motivation
- **Purpose:** Quantify the economic consequences of climate-driven agricultural reorganization in the US at county resolution.
- **Creators:** Keshav Krishnan
- **Funding:** Self-funded research

## Composition
- **Instances:** 6 CSV files covering yield projections, climate projections, stranded assets, decline indicators, insurance mispricing, and opportunity frontier
- **Total records:** ~437,000 across all files
- **Geographic coverage:** 2,902 CONUS counties (excluding AK, HI, PR)
- **Temporal coverage:** Historical (1950-2023), Projections (2025-2050)
- **Crops:** corn, soybeans, winter wheat, spring wheat, cotton, sorghum, barley, oats
- **Scenarios:** SSP2-4.5 (10 GCMs), SSP3-7.0 (9 GCMs)
- **Confidentiality:** No individual-level data. All inputs are publicly available federal datasets.

## Collection Process
- **Source data:** USDA NASS, NOAA nClimDiv, CMIP6 (Pangeo Cloud), USDA RMA, Census ACS, BLS CPI-U
- **Processing:** Deduplication, CPI deflation to 2023 USD, FIPS standardization
- **Models:** LightGBM ensemble yield model (Spearman ρ=0.45, R²=0.21), Ricardian hedonic (R²=0.73), IV/2SLS (F=1,184)
