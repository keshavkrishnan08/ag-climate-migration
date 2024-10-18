"""
Build county-level climate projections from CMIP6 gridded data using the delta method.

Units:
  PRISM baseline  : Fahrenheit (tmax, tmin), mm/month (precip)
  CMIP6 tasmax/min: Kelvin  → convert delta to °F before adding to PRISM
  CMIP6 pr        : kg/m²/s → convert to mm/month, then take delta

Delta method:
  delta = GCM(target_year) - GCM(2025-2030_mean)
