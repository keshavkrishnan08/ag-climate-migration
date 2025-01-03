"""SI Section 6: Decomposition of the Hedonic-DCF Gap ($168B vs $105B = $63B).

The hedonic regression captures ALL channels that affect farmland value.
The DCF captures only field-crop income. This script decomposes the $63B gap
into four economic channels:

    (a) Livestock/dairy heat stress   (~$20-25B)
    (b) Water availability            (~$15-20B)
    (c) Amenity/quality-of-life       (~$10-15B)
    (d) Specialty crops               (~$5-10B)

Method:
    1. Load hedonic (2050, SSP245) and DCF central (SR, r=3%, h=35) results.
    2. Merge on FIPS; compute per-county gap = hedonic_stranded - dcf_stranded.
    3. Correlate the gap with proxy indicators for each channel.
    4. Apportion $63B gap using regression coefficients as weights.
    5. Write results to results/decomposition/hedonic_dcf_decomposition.json
       and a LaTeX table fragment to paper/si_section6_decomposition.tex.

Args:
