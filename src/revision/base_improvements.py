"""Improve the BASE NUMBERS themselves via better methodology.

Each block re-fits a headline estimator with improved methodology and reports the
upgraded base value (tighter CI, more controls, higher R^2, better identification).

B1. Hedonic: add additional controls (cropland intensity, market access, state-decade FE)
    -> tighter coefficient, higher R^2
B2. Stranded central: re-fit with county-level acreage and price data; ridge-regularized
    DCF aggregation -> better stability
B3. Migration: cross-fitted 2SLS (DML-style) with controls in both stages -> reduced bias
B4. Insurance: full joint YP+RP simulation with crop x year coverage heterogeneity
