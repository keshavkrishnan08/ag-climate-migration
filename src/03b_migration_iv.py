"""
Fix 4: IV estimation of farm income -> outmigration elasticity.

Approach: Two-stage least squares with two-way county + year FE.

Instrument: Weather-driven income shock. For each county-year, we compute:
    Z_it = sum_c [ yield_detrended_ict * acres_ic_bar * price_c ] / baseline_income_i
    where yield_detrended = actual yield minus county-crop quadratic trend,
    acres_ic_bar = county-crop mean acres (fixed exposure),
    price_c = national commodity price.
