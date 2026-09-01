# Paired statistics -- `s1_real`

Every comparison is paired by seed (seeds fix the initial design, which is byte-identical across methods), positive favours the reference. `W/T/L` counts seeds; ties are exact zeros and are dropped from the sign test. A comparison is marked **resolved** only when the paired t and the exact sign test *both* clear p<0.05 -- deliberately conservative, because the purpose here is to stop a mean ordering being reported as a finding.

Bootstrap: 10000 paired resamples, seed 0.

## real3d

`n_true = 14`, quantum = 1/n_true = 0.0714, matched |S| = 6 (mean declarations by `zombihop`).

### Matched-|S| recall at |S|=6, vs `zombihop`

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `random` | +0.0857 | [+0.036, +0.136] | 8/1/1 | +3.34 | 0.009 | 0.039 | **yes** |
| `gp_qucb` | +0.0571 | [+0.000, +0.114] | 5/3/2 | +1.81 | 0.104 | 0.453 | no |
| `gp_qlogei` | +0.0286 | [-0.036, +0.100] | 4/2/4 | +0.77 | 0.462 | 1.000 | no |
| `gp_ts` | +0.0214 | [-0.036, +0.086] | 4/3/3 | +0.63 | 0.541 | 1.000 | no |
| `zombihop_nc5` | +0.0357 | [-0.043, +0.086] | 8/1/1 | +0.96 | 0.363 | 0.039 | no |

### Means ± std over seeds

| method | peak_ratio | precision | reached_ratio_final | input_cost | wall_s |
|---|---|---|---|---|---|
| `random` | 0.557 ± 0.111 | 0.557 ± 0.111 | 0.993 ± 0.023 | 1000.199 ± 14.894 | 1.244 ± 0.119 |
| `gp_qucb` | 0.543 ± 0.113 | 0.543 ± 0.113 | 0.914 ± 0.066 | 1473.150 ± 30.132 | 2373.596 ± 249.119 |
| `gp_qlogei` | 0.550 ± 0.107 | 0.550 ± 0.107 | 0.886 ± 0.060 | 1088.612 ± 86.632 | 1960.194 ± 35.655 |
| `gp_ts` | 0.514 ± 0.100 | 0.514 ± 0.100 | 0.650 ± 0.109 | 359.009 ± 39.271 | 210.793 ± 8.063 |
| `zombihop` | 0.271 ± 0.125 | 0.672 ± 0.196 | 0.957 ± 0.060 | 111.639 ± 5.094 | 508.810 ± 138.115 |
| `zombihop_nc5` | 0.179 ± 0.091 | 0.835 ± 0.230 | 0.957 ± 0.037 | 108.039 ± 5.063 | 465.424 ± 133.110 |

### `n_consecutive_converged` 2 vs 5 (declared recall)

mean diff +0.0929 [+0.000, +0.179], W/T/L 7/2/1, t=+1.95 p=0.083, sign p=0.070 -- not resolved.

### Is standard BO worse than uniform random here?

Positive favours `random` (Aleks's prediction).

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `gp_qucb` | -0.0286 | [-0.079, +0.007] | 1/6/3 | -1.18 | 0.269 | 0.625 | no |
| `gp_qlogei` | -0.0571 | [-0.129, +0.021] | 2/2/6 | -1.35 | 0.210 | 0.289 | no |
| `gp_ts` | -0.0643 | [-0.114, -0.014] | 1/3/6 | -2.38 | 0.041 | 0.125 | no |

## real4d

`n_true = 27`, quantum = 1/n_true = 0.0370, matched |S| = 15 (mean declarations by `zombihop`).

### Matched-|S| recall at |S|=15, vs `zombihop`

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `random` | -0.0185 | [-0.070, +0.041] | 4/1/5 | -0.61 | 0.557 | 1.000 | no |
| `gp_qucb` | +0.0407 | [-0.011, +0.093] | 6/1/3 | +1.43 | 0.185 | 0.508 | no |
| `gp_qlogei` | +0.0704 | [+0.015, +0.126] | 7/1/2 | +2.35 | 0.043 | 0.180 | no |
| `gp_ts` | +0.0370 | [-0.041, +0.119] | 4/2/4 | +0.87 | 0.409 | 1.000 | no |
| `zombihop_nc5` | -0.0185 | [-0.059, +0.026] | 3/1/6 | -0.81 | 0.440 | 0.508 | no |

### Means ± std over seeds

| method | peak_ratio | precision | reached_ratio_final | input_cost | wall_s |
|---|---|---|---|---|---|
| `random` | 0.311 ± 0.078 | 0.311 ± 0.078 | 0.574 ± 0.050 | 988.012 ± 10.314 | 0.958 ± 0.064 |
| `gp_qucb` | 0.241 ± 0.068 | 0.241 ± 0.068 | 0.341 ± 0.055 | 1339.553 ± 23.835 | 1819.331 ± 32.332 |
| `gp_qlogei` | 0.215 ± 0.072 | 0.215 ± 0.072 | 0.407 ± 0.068 | 1151.530 ± 39.558 | 2592.489 ± 227.054 |
| `gp_ts` | 0.285 ± 0.106 | 0.285 ± 0.106 | 0.533 ± 0.099 | 727.393 ± 38.535 | 298.825 ± 35.043 |
| `zombihop` | 0.185 ± 0.043 | 0.333 ± 0.077 | 0.507 ± 0.103 | 59.527 ± 5.259 | 657.659 ± 307.398 |
| `zombihop_nc5` | 0.093 ± 0.031 | 0.428 ± 0.136 | 0.589 ± 0.085 | 42.591 ± 6.479 | 423.013 ± 170.634 |

### `n_consecutive_converged` 2 vs 5 (declared recall)

mean diff +0.0926 [+0.059, +0.122], W/T/L 9/1/0, t=+5.51 p=<0.001, sign p=0.004 -- **resolved**.

### Is standard BO worse than uniform random here?

Positive favours `random` (Aleks's prediction).

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `gp_qucb` | +0.0593 | [+0.022, +0.096] | 7/2/1 | +2.95 | 0.016 | 0.070 | no |
| `gp_qlogei` | +0.0889 | [+0.056, +0.122] | 9/1/0 | +5.04 | <0.001 | 0.004 | **yes** |
| `gp_ts` | +0.0556 | [+0.007, +0.104] | 7/1/2 | +2.18 | 0.057 | 0.180 | no |

## real6d

`n_true = 68`, quantum = 1/n_true = 0.0147, matched |S| = 15 (mean declarations by `zombihop`).

### Matched-|S| recall at |S|=15, vs `zombihop`

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `random` | -0.0059 | [-0.018, +0.007] | 3/2/5 | -0.88 | 0.399 | 0.727 | no |
| `gp_qucb` | -0.0059 | [-0.013, +0.001] | 1/5/4 | -1.50 | 0.168 | 0.375 | no |
| `gp_qlogei` | -0.0088 | [-0.022, +0.003] | 2/4/4 | -1.26 | 0.239 | 0.688 | no |
| `gp_ts` | -0.0015 | [-0.013, +0.009] | 3/4/3 | -0.25 | 0.811 | 1.000 | no |
| `zombihop_nc5` | +0.0000 | [-0.010, +0.010] | 3/4/3 | +0.00 | 1.000 | 1.000 | no |

### Means ± std over seeds

| method | peak_ratio | precision | reached_ratio_final | input_cost | wall_s |
|---|---|---|---|---|---|
| `random` | 0.029 ± 0.017 | 0.029 ± 0.017 | 0.026 ± 0.017 | 901.659 ± 7.013 | 0.971 ± 0.113 |
| `gp_qucb` | 0.029 ± 0.016 | 0.029 ± 0.016 | 0.018 ± 0.009 | 1114.443 ± 9.910 | 1754.563 ± 27.034 |
| `gp_qlogei` | 0.034 ± 0.027 | 0.034 ± 0.027 | 0.029 ± 0.028 | 980.681 ± 36.207 | 2508.538 ± 285.877 |
| `gp_ts` | 0.015 ± 0.016 | 0.015 ± 0.016 | 0.021 ± 0.021 | 816.179 ± 7.733 | 444.830 ± 45.640 |
| `zombihop` | 0.007 ± 0.010 | 0.034 ± 0.049 | 0.007 ± 0.012 | 62.202 ± 2.010 | 4783.747 ± 1060.081 |
| `zombihop_nc5` | 0.007 ± 0.010 | 0.051 ± 0.081 | 0.004 ± 0.007 | 60.120 ± 1.912 | 2912.863 ± 564.694 |

### `n_consecutive_converged` 2 vs 5 (declared recall)

mean diff +0.0000 [-0.007, +0.009], W/T/L 2/5/3, t=+0.00 p=1.000, sign p=1.000 -- not resolved.

### Is standard BO worse than uniform random here?

Positive favours `random` (Aleks's prediction).

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `gp_qucb` | +0.0000 | [-0.007, +0.007] | 3/4/3 | +0.00 | 1.000 | 1.000 | no |
| `gp_qlogei` | -0.0029 | [-0.013, +0.007] | 4/2/4 | -0.51 | 0.619 | 1.000 | no |
| `gp_ts` | +0.0044 | [-0.006, +0.015] | 3/6/1 | +0.82 | 0.434 | 0.625 | no |

