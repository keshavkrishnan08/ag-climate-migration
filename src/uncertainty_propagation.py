"""Monte Carlo uncertainty propagation for stranded asset DCF estimate.

Propagates yield model uncertainty (R²=0.21) through the DCF stranded asset
computation using 1,000 Monte Carlo draws from the residual distribution.

Outputs:
    results/stranded_assets/uncertainty_propagation.json
"""

import json
