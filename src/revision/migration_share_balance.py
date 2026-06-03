"""Shift-share identification check (Goldsmith-Pinkham 2020; Borusyak-Hull-Jaravel 2022).
The leave-one-out Bartik instrument is z_it = sum_c share_ic(2000-09 baseline) * price_c *
g_loo_{c,t}. Identification rests either on baseline-share exogeneity or on shock exogeneity
(the shocks are OTHER counties' national crop-yield innovations, plausibly exogenous to a single
county). We test the share-exogeneity leg directly: if baseline shares were correlated with a
county's pre-existing demographic trajectory, the instrument would proxy a pre-trend. So we
regress PRE-PERIOD (pre-estimation-window) population growth on the county's instrument exposure.
A null = no pre-trend = the instrument is not aligned with pre-existing differential trends.
Seed 42; writes only to results/revision/.
"""
import sys; sys.path.insert(0, 'src/revision')
