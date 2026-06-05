"""Revision v4: push anomaly R^2 with additional process-relevant features.

Builds on v3 (VPD, EDD30, heat-days, soil-moisture stress) by adding the thermal
and critical-period precipitation features standard in the modern crop-yield
literature but absent from the original 50-feature set:
  kdd34_growing   : killing degree-days above 34 C (lethal heat, distinct from EDD>30)
  precip_jul_anom : July precipitation anomaly (grain-fill / silking water stress)
  precip_aug_anom : August precipitation anomaly
  dtr_growing_anom: diurnal temperature range anomaly (cloud/heat-stress signal)
  vpd_aug         : August vapour pressure deficit
