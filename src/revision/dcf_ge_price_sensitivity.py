"""E3: DCF general-equilibrium price-feedback sensitivity.

The central DCF holds real commodity prices flat (USDA-ERS 2024a). Climate-driven
global supply contractions plausibly raise relative US crop prices, which would
partially offset capitalised losses. We test the headline at +0.5% per year real
price growth (Hertel et al. 2010 mid-range global GE estimate for major grains
under SSP2-4.5 to 2050) and at +1.0%/yr as an upper bound.

A real price growth rate g compounds over the 25-year DCF horizon. Net present value
of the gross revenue stream scales by approximately (1 + g/(r-g))/(1 + g_base/(r-g_base))
