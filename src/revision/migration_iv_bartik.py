"""Revision: identification-robust migration IV (Reviewer 1 #2).

The original weather IV was criticised on two grounds: (i) external validity
(the yield->migration link should hold only where farming is a large income
share) and (ii) the exclusion restriction (local heat/drought can move people
through amenity or winter-mildness channels, not only farm income; Rappaport
2007). This rebuild answers both:

  * EXTERNAL VALIDITY: estimate only on ERS farming-dependent counties
    (Type_2015_Farming_NO = 1).

  * EXCLUSION RESTRICTION: instrument county farm-income deviation with a
    LEAVE-ONE-OUT SHIFT-SHARE (Bartik) shock built from OTHER counties' national
    crop-specific yield shocks weighted by the county's pre-period crop mix:
        z_it = sum_c  share_ic(baseline) * price_c * g_{c,t}^{-i}
    where g_{c,t}^{-i} is the mean detrended yield anomaly of crop c in year t
    across all counties EXCEPT i. Because the instrument uses other counties'
    growing conditions, it does not carry county i's own local weather, so it
    cannot move county i's migration through a local-amenity channel. We
    additionally control for the county's own winter-minimum-temperature anomaly
    and an amenity indicator, and run an amenity PLACEBO: the same instrument has
    no reduced-form effect on migration in non-farming counties.

  * SUSTAINED SHOCK: treatment is the 3-year moving average of farm-income
    deviation (out-migration responds to persistent decline, not single years).

Seed 42. Writes only to results/revision/.
"""

import json
