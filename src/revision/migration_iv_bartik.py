"""Revision: identification-robust migration IV (Reviewer 1 #2).

The original weather IV was criticised on two grounds: (i) external validity
(the yield->migration link should hold only where farming is a large income
share) and (ii) the exclusion restriction (local heat/drought can move people
through amenity or winter-mildness channels, not only farm income; Rappaport
2007). This rebuild answers both:

  * EXTERNAL VALIDITY: estimate only on ERS farming-dependent counties
    (Type_2015_Farming_NO = 1).
