# `src/revision/` — Reviewer Guide

The scripts in this directory produce every number cited in the revised manuscript. They are
grouped below by status. The `Makefile` at the repo root runs the **headline** and **robustness**
sets in dependency order. **Superseded** scripts are kept for transparency (showing what was
tried and abandoned) but are not part of the production pipeline.

## Headline scripts (cited in the paper)

Each produces a result that maps to a specific number in the manuscript. See `../../REPRODUCE.md`
for the headline-number → script map and `headline_numbers.py` for the auto-verification.

| Script | Produces | Cited at |
|---|---|---|
| `stranded_revision.py`            | DCF conservative + central + grid | abstract; Results §Stranded; SI S10 |
| `stranded_floor_sensitivity.py`   | Floor $1k/$2k sensitivity ($52–68B) | Results §Stranded; SI Part V |
| `hedonic_strengthened.py`         | Soil/irrigation hedonic ($80B); coefficient stability | SI S12, S14 |
| `dcf_ci_fixed.py`                 | Propagated CI [$37, $77]B | Methods; SI S10 |
| `dollar_robustness.py`            | ML vs process ($59B vs $13B) | SI §Substantive E8 |
| `insurance_rolling_aph.py`        | Decomposition $6.6 → $3.7B residual; $1.6B transfer | abstract; Results §4.3; Table 4 |
| `insurance_rp_and_tay.py`         | RP put ($2.6B); TAY range | Results §4.3; SI S11 |
| `insurance_coverage_endogeneity.py` | Acreage-weighted 0.74; endogeneity test | Methods; SI S9 |
| `insurance_sco.py`                | SCO contribution +$0.01B | Methods; Table 4; SI Part V |
| `migration_iv_bartik.py`          | Shift-share IV base + non-farm reduced form | Methods; Results §4.2 |
| `migration_primeage_panel.py`     | Prime-age panel-FE (β=0.024, p=0.005) | Methods; Results §4.2; SI |
| `migration_wildbootstrap.py`      | 5-yr β=0.049; county p=0.001; wild-cluster p=0.0005 | Methods; Results §4.2; SI |
| `migration_share_balance.py`      | Goldsmith-Pinkham/BHJ balance (R²=0.004 within FE) | SI Migration |
| `migration_depop_montecarlo.py`   | $18B central / $22B median / [$11, $38]B 90% CI | Methods §Migration economic cost |
| `migration_fiscal_chain.py`       | Yield → farm-income → land-value long difference | Results §4.2; SI |
| `migration_farmdependent.py`      | 444 farming-dependent county definition | Results §4.2 |
| `recompute_opportunity.py`        | Net farm income $8.1B; per-state | Results §4.4 |
| `yield_v7_spectrum.py`            | Spectrum + SSURGO + irrigation; R²=0.41 | Methods §Yield model; SI S14 |
| `yield_audit_target_decomp.py`    | z-scale vs %-deviation decomposition | SI §Substantive E5 |
| `framework_common_driver.py`      | Common-cause test (3 of 4 channels) | Discussion; SI Framework cohesion |
| `framework_cohesion.py`           | Old chain-test (kept; honest null) | SI Framework cohesion |
| `substantive_experiments.py`      | E1–E9 all in one (re-review responses) | SI §Substantive robustness experiments |
| `headline_numbers.py`             | Consolidates every cited value into HEADLINE_NUMBERS.json | — |

## Robustness scripts (cited as checks, not headlines)

| Script | Purpose |
|---|---|
| `insurance_fast.py`               | Faster variant of `insurance_rolling_aph.py` for sensitivity grids |
| `pull_ssurgo.py`                  | One-time SSURGO pull via USDA Soil Data Access API |
| `fig07_marginal.py`               | Fig. 7B marginal-effects regeneration |

## Superseded scripts (kept for transparency)

These were tried during development and are superseded by the headline scripts above. They are
not run by `make reproduce`. They remain in the repo so reviewers can trace what was abandoned
and why; the manuscript cites only the headline versions.

| Script | Superseded by | Reason |
|---|---|---|
| `migration_iv_v2.py`              | `migration_iv_bartik.py` | Earlier weather-IV; replaced by shift-share |
| `migration_multiiv.py`            | `migration_iv_bartik.py` | Intermediate multi-instrument exploration |
| `migration_primeage.py`           | `migration_primeage_panel.py` | Cross-section; superseded by panel FE |
| `migration_longdiff.py`           | `migration_fiscal_chain.py` | Long-difference fiscal chain (kept as separate scope) |
| `migration_robustness.py`         | `migration_wildbootstrap.py` | Earlier robustness explorations |
| `opportunity_clean.py`            | `recompute_opportunity.py` | Earlier gross-revenue version |
| `yield_v4_morefeatures.py`        | `yield_v7_spectrum.py` | Earlier feature set |
| `yield_v5_percrop.py`             | `yield_v7_spectrum.py` | Earlier per-crop ensemble |
| `yield_v6_skill.py`               | `yield_v7_spectrum.py` | Skill diagnostics; folded in |
| `yield_v7_baseline.py`            | `yield_v7_spectrum.py` | Baseline before spectrum + irrigation |
| `yield_model_v3_features.py`      | `yield_v7_spectrum.py` | Early feature engineering |
| `yield_audit_cotton.py`           | `yield_audit_target_decomp.py` | Cotton-specific audit |
| `yield_audit_cotton_pct.py`       | `yield_audit_target_decomp.py` | Cotton % variant |
| `yield_audit_drought_huber.py`    | (not used) | Huber drought variant |
| `yield_audit_mlp_stack.py`        | (not used) | MLP ensemble exploration |
| `yield_audit_ceiling.py`          | (not used) | Early ceiling demo, now in `yield_v7_spectrum.py` |
| `yield_irr_final.py`              | `yield_v7_spectrum.py` | Irrigation feature folded in |
| `yield_levels_direct.py`          | `yield_audit_target_decomp.py` | Direct-vs-decomposed levels audit |
| `yield_monotonic.py`              | (not used) | Monotonicity check |
| `yield_stack_levels.py`           | `yield_audit_target_decomp.py` | Stacking exploration |
| `yield_uncertainty.py`            | `dcf_ci_fixed.py` | CI propagation; now in DCF CI fixer |
| `yield_v4_morefeatures.py`        | `yield_v7_spectrum.py` | Earlier features |

## Running the pipeline

```bash
# Full revision pipeline (from repo root)
make reproduce        # ~45 min on a 16 GB laptop
make headline         # consolidate cited numbers
make verify           # show cited vs recomputed for every auto-verified number

# Single subsystem
make rev-stranded     # just the stranded scripts
make rev-insurance    # just insurance
make rev-migration    # just migration
make rev-yield        # just yield model
make rev-framework    # common-cause + chain tests
make rev-substantive  # E1-E9 substantive robustness

# Paper build
make revision-paper   # recompile main + SI + response + tracked-changes PDFs
```

## Data inputs

All scripts read from `../../data/raw/` and `../../data/processed/` — see
`../../REPRODUCE.md` for dataset provenance and download links. Raw data is **not** tracked
in git (it is publicly available; see `.gitignore`).

## Outputs

All scripts write to `../../results/revision/*.json` — also not tracked in git (regenerated by
`make reproduce`). The single auditable summary is `../../results/revision/HEADLINE_NUMBERS.json`,
produced by `make headline`.
