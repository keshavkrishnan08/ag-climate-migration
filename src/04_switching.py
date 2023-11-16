"""Phase 3B: Crop switching model — the adaptation model.

Predicts the probability of a county switching from crop A to crop B
given climate and economic conditions.

Architecture (PRD Section 5.2):
    - Binary classifier for each switching pair
    - Outcome: P(switch from corn to sorghum in county c, year t)
    - Features: multi-year temperature trend, relative profitability,
      neighbor counties that already switched, farm debt
    - Implementation: LightGBM classifier
    - Calibration: Platt scaling for valid probabilities
    - CRITICAL: switching probability must be monotone in temp trend
      (hotter → more likely to switch away from heat-sensitive crops)

Switching pairs (from config):
    - [corn, soybeans]
    - [corn, sorghum]
    - [cotton, soybeans]
    - [wheat_winter, wheat_spring]
