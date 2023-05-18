"""
Fix 4: IV estimation of farm income -> outmigration elasticity.

Approach: Two-stage least squares with two-way county + year FE.

Instrument: Weather-driven income shock. For each county-year, we compute:
    Z_it = sum_c [ yield_detrended_ict * acres_ic_bar * price_c ] / baseline_income_i
    where yield_detrended = actual yield minus county-crop quadratic trend,
    acres_ic_bar = county-crop mean acres (fixed exposure),
    price_c = national commodity price.
    This isolates the weather-driven component of farm revenue.

Treatment: Farm income deviation from county baseline.
    D_it = (income_it - income_i_bar) / income_i_bar
    where income_it = sum_c [ yield_ict * acres_ict * price_c * deflator ]

Outcome: Net outmigration rate from population change.
    Y_it = -(pop_t - pop_{t-1}) / pop_{t-1}
    Positive = population loss = net outmigration.

