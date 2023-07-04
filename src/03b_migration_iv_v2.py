"""
Fix 4 (v2): IV estimation — farm income -> outmigration elasticity.
Three new specifications targeting cleaner out-migration measurement.

Approach 1: Net migration from population components
    net_out_migration_rate = -(pop_t - pop_{t-1}) / pop_{t-1}
    BUT cleaned:
      - Remove counties with |pop_change| > 10% (boundary changes)
      - 3-year rolling population change to smooth noise
      - Weighted by baseline population (upweight large counties)

Approach 2: Same-house inverse proxy
    mobility_rate = 1 - (same_house / total_population)
    = (mobility_total - same_house) / total_population
    Interpretation: fraction of current residents who moved into the county
    in the past year. Closely parallels Spec C/B but includes same-county movers.
    Expected sign: positive (income -> more in-migration -> higher mobility rate).
    Note: this is an IN-migration proxy, not pure out-migration, but captures
    the same economic channel. Lower income -> fewer arrivals AND more departures.

