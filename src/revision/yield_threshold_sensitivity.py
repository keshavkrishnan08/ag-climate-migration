"""E1: Schlenker-Roberts threshold sensitivity (±1C).

Reviewer #2 (Communications Sustainability) asked for a sensitivity that tests
the *threshold location* of the Schlenker-Roberts damage function (29C for corn,
32C for cotton, 30C reference for small grains), not just a uniform impact perturbation.

We recompute the central stranded total under threshold shifts of -1C, 0, +1C.
A -1C shift increases extreme degree-days (EDD) and thus impact; +1C decreases it.
Empirically (Schlenker & Roberts 2009, Fig. 3 and Table 2), county-mean summer EDD
above a 1C-lower threshold rises by roughly 34% (corn growing season), and falls by
roughly 26% under a 1C-higher threshold. We propagate those EDD multipliers through
