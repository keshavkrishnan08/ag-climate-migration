"""E2: Single-factor confirmatory SEM for the four-channel common-cause framework.

Reviewer asked for a structural test with explicit fit statistics rather than a
vignette-level unifying claim. We fit a single-factor congeneric model on the
standardised cross-channel residuals (after partialing county fixed effects and
land-value level), with the latent factor F representing forward physical climate
exposure (the common driver established in framework_common_driver.json):

    z_C1 = lambda_1 F + e_1   (stranded value per acre, residualised on land-value)
    z_C2 = lambda_2 F + e_2   (insurance underpricing residual)
