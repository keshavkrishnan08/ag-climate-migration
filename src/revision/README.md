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
