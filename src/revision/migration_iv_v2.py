"""Revision v2: dose-response identification of the farm-income migration channel.

The within-farm-dependent IV (migration_iv_bartik.py) has a strong first stage
(F~66) and the correct sign, but (i) is only marginally significant on the small
farm-dependent subsample and (ii) the simple amenity placebo is not a clean zero
because ERS "non-farming" counties still contain agriculture.

This script uses the stronger, exclusion-robust DOSE-RESPONSE design. Pool all
counties and estimate

