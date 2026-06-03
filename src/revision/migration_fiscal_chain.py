"""R1-2f: the direct chain crop yields -> farm income -> local fiscal capacity,
in farming-dependent counties. Reviewer 1 suggested focusing on this more-defensible
relationship. We estimate, with county + year fixed effects (cluster-robust SE):
  Link 1: farm revenue on lagged county yield (acreage-weighted) -- the income channel.
  Link 2a: farmland value (the agricultural property-tax base) on farm revenue.
  Link 2b: median household income on farm revenue.
The farmland-value link is the fiscal mechanism: ag property tax = assessed land
value x rate, so a farm-income-driven decline in land value contracts the local tax
base (Census 2025). Seed 42."""
import json
