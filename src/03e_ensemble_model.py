"""Phase 3E: Ensemble yield model (LightGBM v2 + Ridge + RandomForest).

The base LightGBM v2 model achieves R²=0.21 on z-scored yield anomalies.
Ensembles routinely add 0.03-0.08 R² by averaging diverse predictors that
each capture different aspects of the yield-climate relationship:
  - LightGBM v2: deep tree, compound drought interactions, threshold effects
  - Ridge:        linear climate signal, regularised against noise
  - RandomForest: bagged trees, robust to outlier years

All three train on IDENTICAL features and splits. The ensemble is a simple
unweighted average (equal weights).

Approaches tested:
  1. Ensemble (LightGBM + Ridge + RF), train ≤ 2012, test 2013-2023
  2. Extended training set (train ≤ 2012, adds 2010-2012 vs prior ≤ 2009 split)
     — already the default in v2; carried forward here.

Temporal splits (match CLAUDE.md Critical Rules):
  Train:  years ≤ 2012 (val_end in config.yaml)
  Test:   2013-2023
