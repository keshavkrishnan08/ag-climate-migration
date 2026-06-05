"""Direct yield-LEVELS model -- the AgMIP/hybrid-ML-comparable metric (R2>0.5 bar).

Reviewer 2's ">0.5 at county scale" benchmark refers to predicting actual yields
(levels), as the GNN/hybrid-ML county papers report. We therefore train a per-crop
model that predicts yield_bu_acre directly from: a technology-time term, the monthly
temperature-exposure spectrum (Schlenker-Roberts degree-time bins), monthly
precipitation/VPD/PDSI, a soil-productivity index (NCCPI proxy) and latitude.
Train <=2012, test 2013-2023; report levels R2 per crop. Also adds a mechanistic
predictor (process-based water-stress-adjusted GDD) so the model is a genuine
mechanistic-ML hybrid. Seed 42.
