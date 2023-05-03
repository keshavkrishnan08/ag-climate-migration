"""Phase 3A: Core yield trend model — LightGBM.

Predicts county-crop yield as a function of climate and technology features.
This is the engine of all projections.

Architecture (PRD Section 5.1):
    - LightGBM gradient boosted trees
    - Outcome: detrended yield anomaly (z-score)
    - After prediction: re-add projected technology trend
    - Final projected yield = tech_component + climate_component
