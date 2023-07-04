"""
Fix 4 (v2): IV estimation — farm income -> outmigration elasticity.
Three new specifications targeting cleaner out-migration measurement.

Approach 1: Net migration from population components
    net_out_migration_rate = -(pop_t - pop_{t-1}) / pop_{t-1}
    BUT cleaned:
      - Remove counties with |pop_change| > 10% (boundary changes)
      - 3-year rolling population change to smooth noise
      - Weighted by baseline population (upweight large counties)
