# Alpha Consistency Report

- timestamp: 2026-04-06T18:24:40.580790+00:00
- threshold: 0.5
- skip_geometric: True
- fast_mode: True

## Summary

- n_functions: 4
- n_consistent: 3
- consistency_rate: 0.750
- max_discrepancy_global: 0.875978
- mean_discrepancy_global: 0.447466

## Per-function Results

| function | alpha_comb | alpha_spec | alpha_geo | discrepancy | consistent | best |
| --- | ---: | ---: | ---: | ---: | :---: | --- |
| poly_cubic | 0.000000 | 0.000000 | nan | 0.000000 | True | geometric |
| sin | 0.419174 | 0.000000 | nan | 0.419174 | True | combinatorial |
| exp | 0.494713 | 0.000000 | nan | 0.494713 | True | combinatorial |
| tanh | 0.875978 | 0.000000 | nan | 0.875978 | False | combinatorial |

This report quantifies empirical alignment between alpha definitions; it is not a formal proof of equivalence.