"""Revision v2: dose-response identification of the farm-income migration channel.

The within-farm-dependent IV (migration_iv_bartik.py) has a strong first stage
(F~66) and the correct sign, but (i) is only marginally significant on the small
farm-dependent subsample and (ii) the simple amenity placebo is not a clean zero
because ERS "non-farming" counties still contain agriculture.

This script uses the stronger, exclusion-robust DOSE-RESPONSE design. Pool all
counties and estimate

    pop_growth_3yr_it = b1 * z_it + b2 * (z_it * farm_intensity_i)
                        + winter_anom_it + county_FE + year_FE + e_it

where z_it is the leave-one-out shift-share national-yield instrument and
farm_intensity_i is the county's pre-period crop-revenue dependence (time
invariant, absorbed by the county FE). The coefficient of interest is b2: the
EXTRA migration response to the farm-income instrument that scales with farm
dependence. Any uniform effect of z that operates through non-farm channels
(amenity, macro commodity cycles) is captured by b1 and differenced out, so b2
isolates the farm-income channel. We confirm with state x year fixed effects
