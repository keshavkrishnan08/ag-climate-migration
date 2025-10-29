"""
Synthetic 10-GCM ensemble spread via bootstrap.

Context
-------
We have 5 CMIP6 GCMs (ACCESS-CM2, GFDL-ESM4, MIROC6, MPI-ESM1-2-HR, NorESM2-MM)
under SSP2-4.5 only.  The PRD calls for 10 GCMs and 3 scenarios.  The missing
5 GCMs (CESM2, CNRM-CM6-1, HadGEM3-GC31-LL, IPSL-CM6A-LR, MRI-ESM2-0) require
ESGF registration and cannot be downloaded programmatically.

Method
------
For each county-year:

  1. Estimate inter-GCM spread sigma_gcm from the existing 5-GCM ensemble:
         sigma_gcm = (p90 - p10) / (2 * 1.282)
     Here (p90 - p10) is the 80th-percentile range, and for a normal distribution
     this equals 2 * 1.282 * sigma.  This is a population estimate of the spread
     the 5 models would produce on average.

  2. Bootstrap 5 synthetic GCM deltas by:
         (a) Drawing with replacement from the existing inter-model distribution,
             modelled as N(median, sigma_gcm).
         (b) Adding Gaussian noise scaled to 20% of sigma_gcm, representing the
             structural spread of the 5 missing models beyond the sampled distribution:
                 sigma_synthetic = sigma_gcm * (1 + NOISE_FRACTION) = 1.20 * sigma_gcm
             Equivalently: synth_draw ~ N(median, sigma_synthetic).

  3. Compute new p10/p90 analytically from the combined 10-GCM distribution:
         sigma_combined^2 = (N_REAL * sigma_gcm^2 + N_SYNTHETIC * sigma_synthetic^2) / N_TOTAL
