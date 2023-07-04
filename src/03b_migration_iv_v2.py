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

Approach 3: Gross out-migration via population accounting
    gross_out_rate = net_outmig_rate + in_migration_rate
    = -(pop_t - pop_{t-1})/pop_{t-1} + true_diff_county_in_rate
    Captures actual outflows regardless of inflow variation.

Summary of all IV specifications:
    Spec A  : net outmigration (raw pop change)              [original, p=0.49]
    Spec A2 : net outmigration cleaned (excl boundary, 3yr)  [Approach 1]
    Spec A3 : net outmigration pop-weighted                   [Approach 1 variant]
    Spec B  : gross mobility rate (1 - same_house/pop)        [Approach 2]
