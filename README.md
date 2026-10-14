# Agricultural Climate Migration

**County-level economics of climate change for US field crops. One forward-looking yield signal,
priced four ways: stranded farmland, crop insurance that lags the climate, rural out-migration,
and new opportunity in northern counties.**

![Dot-and-interval chart of the repo's dollar estimates: stranded farmland value of $52B by discounted cash flow (95% interval $37B to $77B), $80B by a hedonic land-price model and $13B with process-based yields; insurance underpricing of $3.7B a year in 2040 to 2050, or $0.8B with process-based yields; and a northern net farm income gain of $8.1B a year, with a band of $5.5B to $11.1B](docs/channel-estimates.png)

Most climate and agriculture work stops at the yield loss. The cost reaches people through
institutions that look backward. Farmland is priced on past rents. Crop insurance premiums are
set from a ten-year yield history. County budgets follow today's population. This project
follows one projected yield signal from CMIP6 into four dollar channels at county resolution,
reports each with the method that produced it, and then tests whether the four share a single
cause.

```
USDA NASS county yields (1950 to 2023) + PRISM climate + CMIP6 (10 GCMs, SSP2-4.5)
    ->  per-crop LightGBM yield model, 8 field crops (train to 2012, test 2013 to 2023)
    ->  county yield projections through 2050
    ->  stranded farmland value: 30-year DCF at 4% real, cross-checked by hedonic land prices
    ->  insurance: frozen vs rolling APH premiums, 2040 to 2050 (RMA Summary of Business)
    ->  migration: shift-share IV of farm income on prime-age population (Census PEP)
    ->  northern opportunity: warming-driven farm income in 514 counties
    ->  common-driver test across channels  ->  results/revision/HEADLINE_NUMBERS.json
```

The climate moves forward. Land prices, premiums and budgets look back.

## The core idea

1. **One signal, four consequences.** The same forward climate exposure (change in extreme
   degree days, July maximum temperature, projected growing degree days) should drive stranded
   land value, insurance underpricing, rural decline and northern gains, with consistent signs.
2. **Test a common cause, not a chain.** An earlier causal-chain version (stranded value leads to
   decline) failed, so the framework test asks whether one physical exposure variable predicts
   each channel on its own.
3. **Report the method next to the number.** Stranded value comes from a DCF and from a hedonic
   model; yields come from an ML model and a process-based one; migration inference is shown
   under more than one clustering. The spread between them is part of the result.

## What it does (computed by the pipeline, stored in `results/revision/`)

| Channel | Metric | Result |
|---|---|---|
| Stranded farmland value | DCF, SSP2-4.5, 4% real over 30 years, 95% Monte Carlo interval | **$52B** ($37B to $77B) across 1,674 counties |
| Stranded farmland value | Hedonic land prices with soil and irrigation controls (n = 3,004) | $80B |
| Crop insurance underpricing | 2040 to 2050 average, after rolling APH and trend adjustment absorb what they can | **$3.7B per year** (gross $6.6B) |
| Rural out-migration | Farm income on 5-year prime-age population growth, FE-2SLS, 429 farming-dependent counties | beta = 0.049, p = 0.001 (county clusters), p = 0.11 (two-way) |
| Northern opportunity | Net farm income in 514 counties at a 22% margin (15% to 30% band) | **+$8.1B per year** ($5.5B to $11.1B) |
| Common driver | Exposure predicts insurance, decline and opportunity channels, Bonferroni-adjusted | 3 of 3 significant (p = 0.0008, 0.002, 0.014) |
| Yield model | Held-out 2013 to 2023, % deviation from trend: R-squared / Spearman | 0.41 / 0.64 on 52,280 county-crop-years |

Each number is written to a committed JSON file by the script named in
`results/revision/HEADLINE_NUMBERS.json`, and `make verify` rebuilds that file and prints stored
against recomputed values. The strongest results are the insurance gap and the common-driver
test. The migration channel is the weakest, as the Honesty section explains.

## How it works

**Data.** USDA NASS county yields for corn, soybeans, winter and spring wheat, cotton, sorghum,
barley and oats (1950 to 2023), PRISM climate, CMIP6 projections from 10 GCMs via Pangeo with
delta-method downscaling, the RMA Summary of Business, Census ACS and population estimates, the
ERS county typology, and BLS CPI. All dollars are 2023 USD (CPI 304.7). Every source is public.

**Yield model.** Per-crop LightGBM on 60 features: monthly degree-time in temperature bins
(Schlenker-Roberts style), precipitation, vapor pressure deficit, drought index, soil productivity
and latitude. The target is the percentage deviation from each county-crop's linear trend fit on
years up to 2012, and the test set is 2013 to 2023.
Projections apply the 10-GCM SSP2-4.5 ensemble through 2050.

**Stranded value.** Climate-only yield loss times price times harvested acres, discounted at 4%
real over 30 years. The interval propagates independent county error, a state-level spatially
correlated error and GCM spread over 2,000 draws. The hedonic cross-check regresses farmland
value on July maximum temperature with SSURGO soil and irrigation controls (R-squared 0.73); the
warming coefficient moves 5.5% when those controls are added.

**Insurance.** Premiums are computed against a frozen historical APH baseline and against the
projected 2040 to 2050 yield path. A rolling APH window absorbs $2.0B per year and trend-adjusted
yield absorbs $0.85B; the $3.7B per year that remains is the mispricing current rules cannot
remove, of which $1.6B is a cross-subsidy transfer. Acreage-weighted coverage is 0.74.

**Migration.** County and year fixed effects, with a leave-one-out shift-share instrument
(national crop yield shocks times each county's baseline crop mix) for the 3-year moving average
of farm income. Outcome: 5-year growth in the 25 to 54 population. Inference uses 429 county
clusters and a Webb wild-cluster bootstrap with 9,999 draws (p = 0.0001). In the 3-year panel,
the placebo in low-farm-intensity counties is null (p = 0.35).

**Framework test.** One exposure variable per channel, regressed with HC1 errors. Exposure
predicts insurance underpricing, the rural-decline count and northern opportunity, all with the
predicted sign; the first common factor carries 35% of the variance.

## Run it

```bash
conda env create -f environment.yml
conda activate ag_migration
# download raw inputs as described in data/raw/README.md (about 12 GB, not in git)
make pipeline        # steps 1 to 17, ingest through HEADLINE_NUMBERS.json
make verify          # rebuild the headline summary and print stored vs recomputed values
make test            # pytest with an 85% coverage gate
make pipeline-help   # list the individual stages
```

`REPRODUCE.md` maps every reported output to its script and JSON file.

## Layout

```
src/
  01_ingest.py ... 10_figures.py   core pipeline: ingest, features, yield model, switching, projections,
                                   stranded value, migration cascade, insurance, frontier, figures
  03b_ ... 03e_, 06b_, 06c_        alternative migration, yield and hedonic models
  download_*.py, build_*.py        CMIP6, CDL and county projection builders
  revision/                        DCF and hedonic, insurance decomposition, migration IV, yield skill,
                                   framework tests, sensitivity batteries, headline summary
  utils/                           CPI deflator, geography, validation
results/revision/                  committed JSON for every reported number; HEADLINE_NUMBERS.json indexes them
data/raw/README.md                 where to download each input
data/published_dataset/            README and datasheet for the county CSVs (CSVs not committed)
scripts/                           extended-data figure script
tests/                             unit and integration tests
config.yaml                        crops, scenarios, model and valuation parameters
Makefile                           every stage as a target
REPRODUCE.md                       output-to-script map
```

## Honesty

* The yield model moves the dollars. The same DCF with process-based (Schlenker-Roberts) yields
  gives $13B instead of $59B. The two agree on where losses fall (rank correlation 0.66) but
  share only 15% of their top 100 counties. The insurance residual drops from $3.7B to $0.83B
  per year under the process-based yields. The headline numbers rest on the ML yield path.
* Migration is the weakest channel. The 5-year prime-age effect is significant with county
  clusters (p = 0.001) but not with two-way county and year clusters (p = 0.11), which the code
  attributes to having only about 13 year clusters. Nearby specifications are weak too: 3-year
  total population growth in farming-dependent counties has p = 0.099, and the in-migration rate
  has p = 0.87.
* The causal-chain story failed: stranded value does not predict rural decline (p = 0.50). In the
  common-driver test the stranded-value channel is not significant either (p = 0.56) because it
  is confounded by land-value levels.
* Most of the yield model's skill comes from the choice of target. On the Schlenker-Roberts
  z-anomaly target, the new features lift R-squared only from 0.09 to 0.14; switching to
  percentage deviation from trend accounts for most of the 0.41. Cotton is weak (R-squared 0.16).
* Two refinement entries in `results/revision/tier3_tighten.json` do not reconcile with the
  results they refine: a bootstrap interval for the migration coefficient that excludes its own
  point estimate, and a 1M-draw stranded-value interval ($38.6B to $83.4B) that differs from the
  2,000-draw one ($37.4B to $77.2B). Neither is used above.
* Raw inputs, processed parquets, the published county CSVs, figures and the manuscript are not
  in the repo. Some headline values (for example the $61B floored central stranded estimate) are
  computed from local parquets and cannot be rechecked without running the pipeline.
* Scope: CONUS, eight field crops, SSP2-4.5 as the primary scenario. Livestock, specialty crops
  and SSP5-8.5 are not covered.

MIT licensed.
