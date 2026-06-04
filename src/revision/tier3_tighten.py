"""Tier-3: rigorously tighten every cited number.

Strategy:
- Migration wild-cluster bootstrap at B=9999 (vs 1999) -> tighter p-value precision
- Depopulation NPV at 1M draws (vs 200k) -> narrower 90% CI
- Stranded CI propagation at 1M draws
- Hedonic with HC3 robust SEs (more conservative than HC1) + cluster-robust on state
- Migration with HC3 + cluster bootstrap CIs
- Yield acreage-weighted Spearman (the valuation-relevant metric)
- Common-cause test with Bonferroni adjustment across 3 channels
