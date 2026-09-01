# Paired statistics -- `s1_v2`

Every comparison is paired by seed (seeds fix the initial design, which is byte-identical across methods), positive favours the reference. `W/T/L` counts seeds; ties are exact zeros and are dropped from the sign test. A comparison is marked **resolved** only when the paired t and the exact sign test *both* clear p<0.05 -- deliberately conservative, because the purpose here is to stop a mean ordering being reported as a finding.

Bootstrap: 10000 paired resamples, seed 0.

## real3d

`n_true = 14`, quantum = 1/n_true = 0.0714, matched |S| = 5 (mean declarations by `zombihop`).

### Matched-|S| recall at |S|=5, vs `zombihop`

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `random` | +0.0036 | [-0.036, +0.046] | 8/3/9 | +0.16 | 0.874 | 1.000 | no |
| `gp_qucb` | -0.0071 | [-0.093, +0.071] | 5/1/4 | -0.17 | 0.872 | 1.000 | no |
| `gp_qlogei` | -0.0214 | [-0.079, +0.029] | 3/4/3 | -0.71 | 0.496 | 1.000 | no |
| `gp_ts` | -0.0321 | [-0.061, -0.004] | 2/11/7 | -2.13 | 0.046 | 0.180 | no |
| `zombihop_nc5` | +0.0071 | [-0.050, +0.064] | 4/2/4 | +0.23 | 0.823 | 1.000 | no |
| `zombihop_mz0` | -0.0143 | [-0.100, +0.064] | 5/1/4 | -0.32 | 0.758 | 1.000 | no |

### Means ± std over seeds

| method | peak_ratio | precision | reached_ratio_final | input_cost | wall_s |
|---|---|---|---|---|---|
| `random` | 0.589 ± 0.092 | 0.589 ± 0.092 | 0.996 ± 0.016 | 1002.142 ± 14.483 | 1.035 ± 0.231 |
| `gp_qucb` | 0.543 ± 0.113 | 0.543 ± 0.113 | 0.914 ± 0.066 | 1473.150 ± 30.132 | 2373.596 ± 249.119 |
| `gp_qlogei` | 0.550 ± 0.107 | 0.550 ± 0.107 | 0.886 ± 0.060 | 1088.612 ± 86.632 | 1960.194 ± 35.655 |
| `gp_ts` | 0.511 ± 0.096 | 0.511 ± 0.096 | 0.657 ± 0.089 | 367.131 ± 47.699 | 207.551 ± 7.720 |
| `zombihop` | 0.207 ± 0.118 | 0.645 ± 0.272 | 0.939 ± 0.053 | 110.816 ± 5.571 | 470.073 ± 149.536 |
| `zombihop_nc5` | 0.179 ± 0.061 | 0.856 ± 0.206 | 0.971 ± 0.037 | 107.164 ± 5.295 | 453.269 ± 100.466 |
| `zombihop_mz0` | 0.321 ± 0.152 | 0.667 ± 0.190 | 0.971 ± 0.037 | 116.980 ± 4.455 | 671.673 ± 364.494 |

### `n_consecutive_converged` 1 vs 5 (declared recall)

mean diff +0.0571 [-0.014, +0.129], W/T/L 6/2/2, t=+1.56 p=0.153, sign p=0.289 -- not resolved. Values read from the runs themselves, not assumed.

### Is standard BO worse than uniform random here?

Positive favours `random` (Aleks's prediction).

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `gp_qucb` | -0.0500 | [-0.086, -0.014] | 0/5/5 | -2.69 | 0.025 | 0.062 | no |
| `gp_qlogei` | -0.0643 | [-0.129, +0.007] | 2/2/6 | -1.78 | 0.108 | 0.289 | no |
| `gp_ts` | -0.0357 | [-0.071, -0.004] | 3/9/8 | -1.95 | 0.066 | 0.227 | no |

## real4d

`n_true = 27`, quantum = 1/n_true = 0.0370, matched |S| = 11 (mean declarations by `zombihop`).

### Matched-|S| recall at |S|=11, vs `zombihop`

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `random` | -0.0444 | [-0.087, -0.002] | 5/2/13 | -2.01 | 0.059 | 0.096 | no |
| `gp_qucb` | +0.0000 | [-0.048, +0.048] | 5/1/4 | +0.00 | 1.000 | 1.000 | no |
| `gp_qlogei` | +0.0259 | [-0.022, +0.070] | 6/1/3 | +1.02 | 0.333 | 0.508 | no |
| `gp_ts` | +0.0037 | [-0.035, +0.039] | 8/4/8 | +0.19 | 0.850 | 1.000 | no |
| `zombihop_nc5` | -0.0444 | [-0.100, +0.019] | 3/1/6 | -1.36 | 0.206 | 0.508 | no |
| `zombihop_mz0` | -0.0185 | [-0.056, +0.015] | 3/2/5 | -0.92 | 0.381 | 0.727 | no |

### Means ± std over seeds

| method | peak_ratio | precision | reached_ratio_final | input_cost | wall_s |
|---|---|---|---|---|---|
| `random` | 0.333 ± 0.088 | 0.333 ± 0.088 | 0.570 ± 0.084 | 987.703 ± 9.550 | 0.962 ± 0.092 |
| `gp_qucb` | 0.241 ± 0.068 | 0.241 ± 0.068 | 0.341 ± 0.055 | 1339.553 ± 23.835 | 1819.331 ± 32.332 |
| `gp_qlogei` | 0.215 ± 0.072 | 0.215 ± 0.072 | 0.407 ± 0.068 | 1151.530 ± 39.558 | 2592.489 ± 227.054 |
| `gp_ts` | 0.270 ± 0.079 | 0.270 ± 0.079 | 0.494 ± 0.108 | 723.213 ± 45.666 | 284.389 ± 30.003 |
| `zombihop` | 0.148 ± 0.048 | 0.378 ± 0.141 | 0.556 ± 0.086 | 62.238 ± 4.777 | 469.901 ± 211.325 |
| `zombihop_nc5` | 0.093 ± 0.064 | 0.362 ± 0.213 | 0.570 ± 0.074 | 49.006 ± 5.831 | 479.700 ± 238.099 |
| `zombihop_mz0` | 0.148 ± 0.055 | 0.314 ± 0.118 | 0.596 ± 0.088 | 65.740 ± 5.028 | 416.038 ± 231.263 |

### `n_consecutive_converged` 2 vs 5 (declared recall)

mean diff +0.0556 [+0.019, +0.089], W/T/L 7/2/1, t=+2.87 p=0.018, sign p=0.070 -- not resolved. Values read from the runs themselves, not assumed.

### Is standard BO worse than uniform random here?

Positive favours `random` (Aleks's prediction).

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `gp_qucb` | +0.0556 | [+0.026, +0.085] | 7/3/0 | +3.50 | 0.007 | 0.016 | **yes** |
| `gp_qlogei` | +0.0815 | [+0.048, +0.115] | 8/2/0 | +4.49 | 0.002 | 0.008 | **yes** |
| `gp_ts` | +0.0481 | [+0.013, +0.080] | 14/3/3 | +2.73 | 0.013 | 0.013 | **yes** |

## real6d

`n_true = 68`, quantum = 1/n_true = 0.0147, matched |S| = 9 (mean declarations by `zombihop`).

### Matched-|S| recall at |S|=9, vs `zombihop`

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `random` | -0.0029 | [-0.013, +0.007] | 4/1/5 | -0.56 | 0.591 | 1.000 | no |
| `gp_qucb` | -0.0044 | [-0.012, +0.003] | 2/3/5 | -1.15 | 0.279 | 0.453 | no |
| `gp_qlogei` | -0.0015 | [-0.010, +0.009] | 2/5/3 | -0.29 | 0.780 | 1.000 | no |
| `gp_ts` | -0.0015 | [-0.012, +0.009] | 3/3/4 | -0.26 | 0.798 | 1.000 | no |
| `zombihop_nc5` | +0.0044 | [-0.001, +0.010] | 4/5/1 | +1.41 | 0.193 | 0.375 | no |

### Means ± std over seeds

| method | peak_ratio | precision | reached_ratio_final | input_cost | wall_s |
|---|---|---|---|---|---|
| `random` | 0.029 ± 0.017 | 0.029 ± 0.017 | 0.026 ± 0.017 | 901.659 ± 7.013 | 0.971 ± 0.113 |
| `gp_qucb` | 0.029 ± 0.016 | 0.029 ± 0.016 | 0.018 ± 0.009 | 1114.443 ± 9.910 | 1754.563 ± 27.034 |
| `gp_qlogei` | 0.034 ± 0.027 | 0.034 ± 0.027 | 0.029 ± 0.028 | 980.681 ± 36.207 | 2508.538 ± 285.877 |
| `gp_ts` | 0.015 ± 0.016 | 0.015 ± 0.016 | 0.021 ± 0.021 | 816.179 ± 7.733 | 444.830 ± 45.640 |
| `zombihop` | 0.006 ± 0.010 | 0.047 ± 0.085 | 0.013 ± 0.015 | 61.341 ± 2.777 | 2479.751 ± 407.657 |
| `zombihop_nc5` | 0.003 ± 0.006 | 0.031 ± 0.065 | 0.004 ± 0.007 | 60.789 ± 2.397 | 2210.379 ± 423.022 |
| `zombihop_mz0` | -- | -- | -- | -- | -- |

### `n_consecutive_converged` 2 vs 5 (declared recall)

mean diff +0.0029 [-0.003, +0.009], W/T/L 3/6/1, t=+1.00 p=0.343, sign p=0.625 -- not resolved. Values read from the runs themselves, not assumed.

### Is standard BO worse than uniform random here?

Positive favours `random` (Aleks's prediction).

| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |
|---|---|---|---|---|---|---|---|
| `gp_qucb` | -0.0015 | [-0.009, +0.006] | 1/6/3 | -0.36 | 0.726 | 0.625 | no |
| `gp_qlogei` | +0.0015 | [-0.006, +0.009] | 4/3/3 | +0.36 | 0.726 | 1.000 | no |
| `gp_ts` | +0.0015 | [-0.007, +0.009] | 4/4/2 | +0.32 | 0.758 | 0.688 | no |

