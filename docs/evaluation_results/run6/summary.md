# Results of run 6, from summarise_evaluations.py

**Same episodes per scenario:** large_plumes yes, one_source yes, small_plumes yes, sources_1_3 yes

## sources_1_3: sources [1, 3], plume width [2.0, 5.0]

### Surveys

| Survey | Actions | Seeds | Ended by goal (%) | AUV steps | Overlap plume | Overlap border | Visited plume (%) | Visited border (%) | Plumes found (%) | Turns |
|---|---|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1 | 100.0 | 370.1 | 0.997 | 0.992 | 98.5 | 96.8 | 99.9 | 34.8 |
| `lawnmower_sweep3_estimate` | scripted | 1 | 100.0 | 281.6 | 0.994 | 0.983 | 84.6 | 84.1 | 100.0 | 25.2 |
| `lawnmower_sweep4_path_end` | scripted | 1 | 100.0 | 216.3 | 0.973 | 0.941 | 74.8 | 74.4 | 100.0 | 18.1 |
| `minimal_values` | deterministic | 1 | 72.8 | 374.1 | 0.966 | 0.952 | 92.1 | 90.8 | 96.2 | 134.0 |
| `minimal_values` | sampled | 1 | 84.6 | 374.0 | 0.991 | 0.982 | 95.2 | 94.2 | 98.8 | 113.5 |
| `random_sweep4_handover_visit` | scripted | 1 | 0.4 | 479.9 | 0.814 | 0.803 | 78.4 | 77.2 | 79.6 | 252.8 |
| `random_sweep6_handover_estimate` | scripted | 1 | 0.5 | 479.7 | 0.867 | 0.858 | 83.1 | 82.1 | 85.2 | 261.2 |
| `random_trained_visit` | scripted | 1 | 12.6 | 474.2 | 0.957 | 0.939 | 87.4 | 86.3 | 97.1 | 239.4 |
| `rule_space_handover_estimate` | scripted | 1 | 88.1 | 338.7 | 0.995 | 0.988 | 92.0 | 94.8 | 99.6 | 91.8 |
| `rule_sweep4_handover_visit` | scripted | 1 | 99.1 | 304.1 | 0.998 | 0.994 | 96.9 | 96.0 | 100.0 | 78.8 |
| `rule_sweep4_plume_border_visit` | scripted | 1 | 74.8 | 369.1 | 0.987 | 0.976 | 93.3 | 92.3 | 99.1 | 125.4 |
| `rule_sweep4_scripted_gas_visit` | scripted | 1 | 99.9 | 277.3 | 0.998 | 0.992 | 98.1 | 96.4 | 100.0 | 41.2 |
| `rule_sweep6_handover_estimate` | scripted | 1 | 99.9 | 230.7 | 0.997 | 0.990 | 93.3 | 94.0 | 100.0 | 61.1 |
| `rule_sweep6_plume_border_estimate` | scripted | 1 | 89.5 | 290.6 | 0.995 | 0.984 | 92.4 | 93.1 | 99.7 | 100.3 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 1 | 100.0 | 226.5 | 0.997 | 0.989 | 92.8 | 93.1 | 100.0 | 34.2 |
| `rule_trained_visit` | scripted | 1 | 74.9 | 377.4 | 0.985 | 0.971 | 93.0 | 92.8 | 98.5 | 139.1 |
| `space_est` | deterministic | 3 | 78.1 ± 2.0 | 358.0 ± 3.3 | 0.975 ± 0.004 | 0.963 ± 0.004 | 88.1 ± 0.9 | 94.0 ± 0.4 | 97.1 ± 0.7 | 117.9 ± 4.5 |
| `space_est` | sampled | 3 | 86.6 ± 0.6 | 350.1 ± 2.3 | 0.995 ± 0.001 | 0.989 ± 0.001 | 93.9 ± 0.3 | 95.5 ± 0.0 | 99.6 ± 0.2 | 94.5 ± 1.2 |
| `space_only_visit` | scripted | 1 | 83.0 | 399.8 | 0.995 | 0.988 | 91.8 | 91.2 | 99.9 | 70.6 |
| `sweep4_visit` | deterministic | 3 | 98.3 ± 0.2 | 298.1 ± 0.2 | 0.994 ± 0.001 | 0.988 ± 0.001 | 96.7 ± 0.1 | 95.9 ± 0.1 | 99.7 ± 0.0 | 76.5 ± 0.2 |
| `sweep4_visit` | sampled | 3 | 99.0 ± 0.2 | 296.1 ± 1.0 | 0.999 ± 0.000 | 0.994 ± 0.000 | 97.1 ± 0.1 | 96.2 ± 0.0 | 100.0 ± 0.0 | 72.9 ± 0.7 |
| `sweep6_est` | deterministic | 3 | 99.9 ± 0.0 | 224.9 ± 1.2 | 0.998 ± 0.000 | 0.991 ± 0.000 | 93.8 ± 0.4 | 94.7 ± 0.1 | 100.0 ± 0.0 | 57.9 ± 0.7 |
| `sweep6_est` | sampled | 3 | 100.0 ± 0.0 | 225.3 ± 1.3 | 0.998 ± 0.000 | 0.991 ± 0.000 | 93.7 ± 0.3 | 94.7 ± 0.1 | 100.0 ± 0.0 | 58.1 ± 1.0 |
| `sweep6_est_control` | deterministic | 3 | 90.1 ± 2.0 | 269.1 ± 3.5 | 0.979 ± 0.002 | 0.961 ± 0.004 | 89.0 ± 0.2 | 91.9 ± 0.7 | 98.6 ± 0.2 | 101.7 ± 4.6 |
| `sweep6_est_control` | sampled | 3 | 99.7 ± 0.1 | 247.6 ± 1.4 | 0.998 ± 0.000 | 0.989 ± 0.000 | 94.9 ± 0.2 | 94.3 ± 0.1 | 100.0 ± 0.0 | 72.9 ± 1.1 |
| `sweep6_est_map` | deterministic | 3 | 99.0 ± 0.3 | 245.0 ± 4.7 | 0.992 ± 0.004 | 0.977 ± 0.004 | 92.5 ± 1.8 | 81.9 ± 1.1 | 99.7 ± 0.4 | 51.6 ± 2.5 |
| `sweep6_est_map` | sampled | 3 | 100.0 ± 0.0 | 241.7 ± 4.7 | 0.996 ± 0.000 | 0.983 ± 0.001 | 92.4 ± 1.6 | 83.6 ± 0.7 | 100.0 ± 0.0 | 50.1 ± 1.4 |
| `sweep6_est_no_time` | deterministic | 3 | 99.8 ± 0.0 | 226.7 ± 0.8 | 0.997 ± 0.000 | 0.990 ± 0.000 | 93.7 ± 0.2 | 94.8 ± 0.1 | 99.9 ± 0.0 | 58.9 ± 0.4 |
| `sweep6_est_no_time` | sampled | 3 | 100.0 ± 0.0 | 227.0 ± 0.7 | 0.998 ± 0.000 | 0.991 ± 0.000 | 93.9 ± 0.1 | 94.9 ± 0.2 | 100.0 ± 0.0 | 59.3 ± 0.4 |
| `sweep6_est_one` | deterministic | 3 | 99.8 ± 0.0 | 227.0 ± 2.6 | 0.998 ± 0.000 | 0.990 ± 0.001 | 92.9 ± 0.4 | 95.1 ± 0.6 | 100.0 ± 0.0 | 59.4 ± 1.9 |
| `sweep6_est_one` | sampled | 3 | 99.9 ± 0.0 | 227.5 ± 2.9 | 0.998 ± 0.000 | 0.991 ± 0.000 | 93.2 ± 0.4 | 95.2 ± 0.5 | 100.0 ± 0.0 | 59.7 ± 2.0 |
| `untuned` | deterministic | 1 | 35.0 | 432.1 | 0.873 | 0.851 | 82.1 | 79.8 | 80.7 | 241.5 |
| `untuned` | sampled | 1 | 74.7 | 405.3 | 0.994 | 0.985 | 96.1 | 94.6 | 99.3 | 147.0 |

### Steps and plumes found per number of true plumes

| Survey | Actions | Steps, 1 plume(s) | Found (%), 1 plume(s) | Steps, 2 plume(s) | Found (%), 2 plume(s) | Steps, 3 plume(s) | Found (%), 3 plume(s) |
|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 368 | 100.0 | 373 | 99.9 | 372 | 100.0 |
| `lawnmower_sweep3_estimate` | scripted | 282 | 100.0 | 281 | 100.0 | 280 | 100.0 |
| `lawnmower_sweep4_path_end` | scripted | 217 | 100.0 | 216 | 100.0 | 216 | 100.0 |
| `minimal_values` | deterministic | 351 | 99.6 | 394 | 96.8 | 410 | 89.7 |
| `minimal_values` | sampled | 356 | 99.8 | 388 | 99.2 | 404 | 96.3 |
| `random_sweep4_handover_visit` | scripted | 480 | 85.5 | 480 | 77.9 | 480 | 74.9 |
| `random_sweep6_handover_estimate` | scripted | 480 | 89.7 | 480 | 84.3 | 480 | 80.6 |
| `random_trained_visit` | scripted | 472 | 97.2 | 476 | 97.9 | 476 | 95.2 |
| `rule_space_handover_estimate` | scripted | 331 | 99.8 | 350 | 99.3 | 334 | 100.0 |
| `rule_sweep4_handover_visit` | scripted | 287 | 100.0 | 318 | 100.0 | 330 | 100.0 |
| `rule_sweep4_plume_border_visit` | scripted | 347 | 99.8 | 389 | 98.4 | 396 | 99.4 |
| `rule_sweep4_scripted_gas_visit` | scripted | 267 | 100.0 | 286 | 100.0 | 296 | 100.0 |
| `rule_sweep6_handover_estimate` | scripted | 214 | 100.0 | 244 | 100.0 | 258 | 100.0 |
| `rule_sweep6_plume_border_estimate` | scripted | 271 | 100.0 | 305 | 99.7 | 326 | 99.1 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 209 | 100.0 | 240 | 100.0 | 257 | 100.0 |
| `rule_trained_visit` | scripted | 364 | 99.8 | 391 | 97.9 | 391 | 97.7 |
| `space_est` | deterministic | 339 ± 2 | 99.6 ± 0.0 | 373 ± 4 | 97.5 ± 0.5 | 391 ± 8 | 92.8 ± 2.0 |
| `space_est` | sampled | 337 ± 3 | 99.8 ± 0.0 | 361 ± 2 | 99.6 ± 0.2 | 368 ± 3 | 99.5 ± 0.7 |
| `space_only_visit` | scripted | 392 | 99.8 | 408 | 99.9 | 407 | 100.0 |
| `sweep4_visit` | deterministic | 282 ± 0 | 100.0 ± 0.0 | 310 ± 0 | 99.6 ± 0.1 | 328 ± 1 | 99.4 ± 0.0 |
| `sweep4_visit` | sampled | 280 ± 1 | 100.0 ± 0.0 | 307 ± 1 | 100.0 ± 0.0 | 328 ± 4 | 100.0 ± 0.0 |
| `sweep6_est` | deterministic | 206 ± 1 | 100.0 ± 0.0 | 238 ± 1 | 100.0 ± 0.0 | 260 ± 2 | 100.0 ± 0.0 |
| `sweep6_est` | sampled | 207 ± 1 | 100.0 ± 0.0 | 239 ± 1 | 100.0 ± 0.0 | 259 ± 3 | 100.0 ± 0.0 |
| `sweep6_est_control` | deterministic | 252 ± 3 | 100.0 ± 0.0 | 283 ± 4 | 97.9 ± 0.3 | 296 ± 7 | 98.2 ± 0.6 |
| `sweep6_est_control` | sampled | 228 ± 1 | 100.0 ± 0.0 | 263 ± 2 | 100.0 ± 0.0 | 281 ± 4 | 99.9 ± 0.1 |
| `sweep6_est_map` | deterministic | 230 ± 4 | 99.9 ± 0.1 | 256 ± 5 | 99.6 ± 0.6 | 273 ± 6 | 99.7 ± 0.4 |
| `sweep6_est_map` | sampled | 224 ± 5 | 100.0 ± 0.0 | 255 ± 5 | 100.0 ± 0.0 | 275 ± 6 | 100.0 ± 0.0 |
| `sweep6_est_no_time` | deterministic | 208 ± 0 | 100.0 ± 0.0 | 241 ± 1 | 99.9 ± 0.0 | 262 ± 1 | 100.0 ± 0.0 |
| `sweep6_est_no_time` | sampled | 209 ± 0 | 100.0 ± 0.0 | 241 ± 1 | 100.0 ± 0.0 | 261 ± 1 | 100.0 ± 0.0 |
| `sweep6_est_one` | deterministic | 208 ± 2 | 100.0 ± 0.0 | 241 ± 3 | 99.9 ± 0.1 | 263 ± 5 | 100.0 ± 0.0 |
| `sweep6_est_one` | sampled | 209 ± 2 | 100.0 ± 0.0 | 241 ± 3 | 100.0 ± 0.0 | 261 ± 4 | 100.0 ± 0.0 |
| `untuned` | deterministic | 414 | 99.4 | 448 | 75.9 | 459 | 64.4 |
| `untuned` | sampled | 385 | 100.0 | 423 | 99.3 | 437 | 98.0 |

### Each seed against the lawnmower and the rule

| Variant | Actions | Seed | Against | Difference in AUV steps (95 % interval) | Ended (%) | Overlap plume / border | Plumes found (%) | Beats it |
|---|---|---|---|---|---|---|---|---|
| `space_est` | deterministic | 1 | `lawnmower_sweep3_estimate` | +73.2 (+67.5 to +78.5) | 79.9 | 0.978 / 0.967 | 97.7 | no |
| `space_est` | deterministic | 1 | `rule_space_handover_estimate` | +16.2 (+9.9 to +22.3) | 79.9 | 0.978 / 0.967 | 97.7 | no |
| `space_est` | deterministic | 2 | `lawnmower_sweep3_estimate` | +74.8 (+69.4 to +80.2) | 79.1 | 0.977 / 0.966 | 97.5 | no |
| `space_est` | deterministic | 2 | `rule_space_handover_estimate` | +17.7 (+11.8 to +23.3) | 79.1 | 0.977 / 0.966 | 97.5 | no |
| `space_est` | deterministic | 3 | `lawnmower_sweep3_estimate` | +80.9 (+75.1 to +86.5) | 75.3 | 0.969 / 0.957 | 96.2 | no |
| `space_est` | deterministic | 3 | `rule_space_handover_estimate` | +23.9 (+17.6 to +29.9) | 75.3 | 0.969 / 0.957 | 96.2 | no |
| `space_est` | sampled | 1 | `lawnmower_sweep3_estimate` | +65.4 (+60.0 to +71.0) | 87.4 | 0.994 / 0.987 | 99.3 | no |
| `space_est` | sampled | 1 | `rule_space_handover_estimate` | +8.4 (+2.6 to +13.8) | 87.4 | 0.994 / 0.987 | 99.3 | no |
| `space_est` | sampled | 2 | `lawnmower_sweep3_estimate` | +71.0 (+65.9 to +76.3) | 86.3 | 0.996 / 0.990 | 99.9 | no |
| `space_est` | sampled | 2 | `rule_space_handover_estimate` | +13.9 (+8.5 to +19.1) | 86.3 | 0.996 / 0.990 | 99.9 | no |
| `space_est` | sampled | 3 | `lawnmower_sweep3_estimate` | +68.9 (+63.6 to +73.9) | 86.1 | 0.996 / 0.990 | 99.7 | no |
| `space_est` | sampled | 3 | `rule_space_handover_estimate` | +11.9 (+6.5 to +17.1) | 86.1 | 0.996 / 0.990 | 99.7 | no |
| `sweep4_visit` | deterministic | 1 | `lawnmower_sweep2_visit` | -71.8 (-74.6 to -69.1) | 98.1 | 0.995 / 0.989 | 99.8 | no |
| `sweep4_visit` | deterministic | 1 | `rule_sweep4_handover_visit` | -5.9 (-8.5 to -3.2) | 98.1 | 0.995 / 0.989 | 99.8 | no |
| `sweep4_visit` | deterministic | 2 | `lawnmower_sweep2_visit` | -72.2 (-75.0 to -69.5) | 98.6 | 0.994 / 0.988 | 99.7 | no |
| `sweep4_visit` | deterministic | 2 | `rule_sweep4_handover_visit` | -6.3 (-8.9 to -3.5) | 98.6 | 0.994 / 0.988 | 99.7 | no |
| `sweep4_visit` | deterministic | 3 | `lawnmower_sweep2_visit` | -72.0 (-74.9 to -69.3) | 98.1 | 0.993 / 0.986 | 99.7 | no |
| `sweep4_visit` | deterministic | 3 | `rule_sweep4_handover_visit` | -6.1 (-8.7 to -3.2) | 98.1 | 0.993 / 0.986 | 99.7 | no |
| `sweep4_visit` | sampled | 1 | `lawnmower_sweep2_visit` | -72.8 (-75.7 to -70.0) | 99.0 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 1 | `rule_sweep4_handover_visit` | -6.9 (-9.4 to -4.4) | 99.0 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 2 | `lawnmower_sweep2_visit` | -75.2 (-77.8 to -72.7) | 99.3 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 2 | `rule_sweep4_handover_visit` | -9.3 (-11.7 to -7.0) | 99.3 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 3 | `lawnmower_sweep2_visit` | -74.0 (-76.8 to -71.4) | 98.8 | 0.999 / 0.994 | 100.0 | no |
| `sweep4_visit` | sampled | 3 | `rule_sweep4_handover_visit` | -8.0 (-10.6 to -5.5) | 98.8 | 0.999 / 0.994 | 100.0 | no |
| `sweep6_est` | deterministic | 1 | `lawnmower_sweep3_estimate` | -58.4 (-61.0 to -55.7) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | deterministic | 1 | `rule_sweep6_handover_estimate` | -7.5 (-8.9 to -6.1) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | deterministic | 2 | `lawnmower_sweep3_estimate` | -56.5 (-59.3 to -53.7) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | deterministic | 2 | `rule_sweep6_handover_estimate` | -5.6 (-7.1 to -4.2) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | deterministic | 3 | `lawnmower_sweep3_estimate` | -55.5 (-58.2 to -52.7) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | deterministic | 3 | `rule_sweep6_handover_estimate` | -4.6 (-6.0 to -3.2) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | sampled | 1 | `lawnmower_sweep3_estimate` | -58.1 (-60.7 to -55.5) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | sampled | 1 | `rule_sweep6_handover_estimate` | -7.2 (-8.6 to -6.0) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | sampled | 2 | `lawnmower_sweep3_estimate` | -55.5 (-58.2 to -52.7) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | sampled | 2 | `rule_sweep6_handover_estimate` | -4.6 (-6.0 to -3.2) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | sampled | 3 | `lawnmower_sweep3_estimate` | -55.4 (-58.0 to -52.5) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est` | sampled | 3 | `rule_sweep6_handover_estimate` | -4.5 (-5.8 to -3.1) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_control` | deterministic | 1 | `lawnmower_sweep3_estimate` | -12.7 (-18.5 to -7.0) | 90.9 | 0.978 / 0.961 | 98.4 | no |
| `sweep6_est_control` | deterministic | 1 | `rule_sweep6_handover_estimate` | +38.2 (+33.3 to +43.2) | 90.9 | 0.978 / 0.961 | 98.4 | no |
| `sweep6_est_control` | deterministic | 2 | `lawnmower_sweep3_estimate` | -16.7 (-22.4 to -10.9) | 92.0 | 0.982 / 0.966 | 98.9 | no |
| `sweep6_est_control` | deterministic | 2 | `rule_sweep6_handover_estimate` | +34.2 (+29.4 to +39.2) | 92.0 | 0.982 / 0.966 | 98.9 | no |
| `sweep6_est_control` | deterministic | 3 | `lawnmower_sweep3_estimate` | -8.2 (-14.7 to -1.9) | 87.4 | 0.977 / 0.957 | 98.5 | no |
| `sweep6_est_control` | deterministic | 3 | `rule_sweep6_handover_estimate` | +42.7 (+37.3 to +48.5) | 87.4 | 0.977 / 0.957 | 98.5 | no |
| `sweep6_est_control` | sampled | 1 | `lawnmower_sweep3_estimate` | -33.4 (-36.8 to -30.1) | 99.7 | 0.998 / 0.989 | 100.0 | yes |
| `sweep6_est_control` | sampled | 1 | `rule_sweep6_handover_estimate` | +17.5 (+15.6 to +19.3) | 99.7 | 0.998 / 0.989 | 100.0 | no |
| `sweep6_est_control` | sampled | 2 | `lawnmower_sweep3_estimate` | -36.1 (-39.3 to -32.7) | 99.9 | 0.998 / 0.989 | 100.0 | yes |
| `sweep6_est_control` | sampled | 2 | `rule_sweep6_handover_estimate` | +14.8 (+12.9 to +16.7) | 99.9 | 0.998 / 0.989 | 100.0 | no |
| `sweep6_est_control` | sampled | 3 | `lawnmower_sweep3_estimate` | -32.7 (-36.1 to -29.2) | 99.6 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est_control` | sampled | 3 | `rule_sweep6_handover_estimate` | +18.2 (+16.1 to +20.4) | 99.6 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est_map` | deterministic | 1 | `lawnmower_sweep3_estimate` | -32.9 (-35.8 to -29.8) | 98.9 | 0.994 / 0.979 | 100.0 | no |
| `sweep6_est_map` | deterministic | 1 | `rule_sweep6_handover_estimate` | +18.0 (+15.4 to +20.7) | 98.9 | 0.994 / 0.979 | 100.0 | no |
| `sweep6_est_map` | deterministic | 2 | `lawnmower_sweep3_estimate` | -33.7 (-36.9 to -30.5) | 99.4 | 0.994 / 0.981 | 100.0 | no |
| `sweep6_est_map` | deterministic | 2 | `rule_sweep6_handover_estimate` | +17.2 (+14.6 to +20.0) | 99.4 | 0.994 / 0.981 | 100.0 | no |
| `sweep6_est_map` | deterministic | 3 | `lawnmower_sweep3_estimate` | -43.3 (-46.8 to -39.7) | 98.6 | 0.986 / 0.972 | 99.2 | no |
| `sweep6_est_map` | deterministic | 3 | `rule_sweep6_handover_estimate` | +7.6 (+4.8 to +10.6) | 98.6 | 0.986 / 0.972 | 99.2 | no |
| `sweep6_est_map` | sampled | 1 | `lawnmower_sweep3_estimate` | -38.2 (-40.9 to -35.7) | 100.0 | 0.996 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 1 | `rule_sweep6_handover_estimate` | +12.7 (+10.7 to +14.7) | 100.0 | 0.996 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 2 | `lawnmower_sweep3_estimate` | -35.2 (-38.1 to -32.4) | 100.0 | 0.996 / 0.984 | 100.0 | yes |
| `sweep6_est_map` | sampled | 2 | `rule_sweep6_handover_estimate` | +15.7 (+13.4 to +17.9) | 100.0 | 0.996 / 0.984 | 100.0 | no |
| `sweep6_est_map` | sampled | 3 | `lawnmower_sweep3_estimate` | -46.3 (-49.3 to -43.2) | 100.0 | 0.996 / 0.983 | 100.0 | yes |
| `sweep6_est_map` | sampled | 3 | `rule_sweep6_handover_estimate` | +4.6 (+2.4 to +6.7) | 100.0 | 0.996 / 0.983 | 100.0 | no |
| `sweep6_est_no_time` | deterministic | 1 | `lawnmower_sweep3_estimate` | -56.0 (-58.8 to -53.1) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_no_time` | deterministic | 1 | `rule_sweep6_handover_estimate` | -5.1 (-6.6 to -3.5) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_no_time` | deterministic | 2 | `lawnmower_sweep3_estimate` | -54.0 (-56.8 to -51.2) | 99.8 | 0.998 / 0.990 | 99.9 | no |
| `sweep6_est_no_time` | deterministic | 2 | `rule_sweep6_handover_estimate` | -3.1 (-4.7 to -1.6) | 99.8 | 0.998 / 0.990 | 99.9 | no |
| `sweep6_est_no_time` | deterministic | 3 | `lawnmower_sweep3_estimate` | -54.9 (-57.8 to -52.0) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_no_time` | deterministic | 3 | `rule_sweep6_handover_estimate` | -4.1 (-5.6 to -2.4) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_no_time` | sampled | 1 | `lawnmower_sweep3_estimate` | -55.4 (-58.1 to -52.5) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_no_time` | sampled | 1 | `rule_sweep6_handover_estimate` | -4.5 (-5.9 to -3.0) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_no_time` | sampled | 2 | `lawnmower_sweep3_estimate` | -53.6 (-56.3 to -50.8) | 100.0 | 0.998 / 0.992 | 100.0 | yes |
| `sweep6_est_no_time` | sampled | 2 | `rule_sweep6_handover_estimate` | -2.7 (-4.1 to -1.3) | 100.0 | 0.998 / 0.992 | 100.0 | yes |
| `sweep6_est_no_time` | sampled | 3 | `lawnmower_sweep3_estimate` | -54.8 (-57.6 to -52.0) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_no_time` | sampled | 3 | `rule_sweep6_handover_estimate` | -3.9 (-5.4 to -2.6) | 100.0 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 1 | `lawnmower_sweep3_estimate` | -55.0 (-57.8 to -52.2) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_one` | deterministic | 1 | `rule_sweep6_handover_estimate` | -4.1 (-5.6 to -2.5) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_one` | deterministic | 2 | `lawnmower_sweep3_estimate` | -57.6 (-60.3 to -54.7) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_one` | deterministic | 2 | `rule_sweep6_handover_estimate` | -6.7 (-8.2 to -5.2) | 99.8 | 0.997 / 0.990 | 99.9 | no |
| `sweep6_est_one` | deterministic | 3 | `lawnmower_sweep3_estimate` | -51.2 (-54.1 to -48.2) | 99.8 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 3 | `rule_sweep6_handover_estimate` | -0.3 (-1.9 to +1.3) | 99.8 | 0.998 / 0.991 | 100.0 | no |
| `sweep6_est_one` | sampled | 1 | `lawnmower_sweep3_estimate` | -54.0 (-56.9 to -51.1) | 99.9 | 0.998 / 0.992 | 100.0 | yes |
| `sweep6_est_one` | sampled | 1 | `rule_sweep6_handover_estimate` | -3.1 (-4.6 to -1.6) | 99.9 | 0.998 / 0.992 | 100.0 | yes |
| `sweep6_est_one` | sampled | 2 | `lawnmower_sweep3_estimate` | -57.7 (-60.4 to -54.9) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_one` | sampled | 2 | `rule_sweep6_handover_estimate` | -6.8 (-8.4 to -5.3) | 99.9 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_one` | sampled | 3 | `lawnmower_sweep3_estimate` | -50.6 (-53.5 to -47.6) | 99.8 | 0.998 / 0.991 | 100.0 | yes |
| `sweep6_est_one` | sampled | 3 | `rule_sweep6_handover_estimate` | +0.3 (-1.2 to +1.8) | 99.8 | 0.998 / 0.991 | 100.0 | no |

### Paired differences in AUV steps, each side the mean over its seeds

| Survey | Against | All episodes | 1 plume | 2 plumes | 3 plumes |
|---|---|---|---|---|---|
| `sweep6_est`, sampled | `rule_sweep6_handover_estimate`, scripted | -5.4 (-6.6 to -4.2) | -7.2 (-8.8 to -5.7) | -5.0 (-7.3 to -3.0) | +0.9 (-2.8 to +4.4) |
| `sweep6_est`, sampled | `rule_sweep6_scripted_gas_estimate`, scripted | -1.1 (-2.3 to +0.1) | -2.2 (-3.9 to -0.7) | -0.7 (-2.7 to +1.1) | +2.2 (-1.6 to +5.9) |
| `sweep6_est`, sampled | `lawnmower_sweep3_estimate`, scripted | -56.3 (-58.8 to -53.6) | -75.3 (-78.9 to -71.7) | -41.9 (-45.5 to -38.2) | -21.4 (-27.6 to -15.3) |
| `sweep6_est`, sampled | `sweep6_est_control`, sampled | -22.2 (-23.7 to -20.8) | -20.8 (-23.0 to -18.8) | -24.1 (-26.4 to -21.9) | -22.1 (-26.2 to -18.0) |
| `sweep6_est`, deterministic | `sweep6_est_control`, deterministic | -44.3 (-48.8 to -39.8) | -46.1 (-53.0 to -39.5) | -44.6 (-51.6 to -37.0) | -35.5 (-47.2 to -23.7) |
| `sweep6_est`, sampled | `sweep6_est_map`, sampled | -16.4 (-17.9 to -14.9) | -17.0 (-19.1 to -14.9) | -15.6 (-17.8 to -13.5) | -16.2 (-21.1 to -11.3) |
| `sweep6_est`, sampled | `sweep6_est_no_time`, sampled | -1.7 (-2.3 to -1.1) | -1.4 (-2.3 to -0.5) | -2.0 (-2.9 to -1.2) | -2.2 (-4.5 to +0.3) |
| `sweep6_est`, sampled | `sweep6_est_one`, sampled | -2.2 (-2.9 to -1.5) | -2.3 (-3.4 to -1.2) | -2.0 (-3.1 to -1.0) | -2.4 (-4.5 to -0.0) |
| `sweep6_est`, deterministic | `sweep6_est`, sampled | -0.5 (-1.0 to +0.2) | -0.8 (-1.6 to +0.1) | -0.6 (-1.5 to +0.3) | +1.4 (-1.2 to +3.8) |
| `sweep6_est_control`, deterministic | `sweep6_est_control`, sampled | +21.6 (+17.2 to +26.1) | +24.4 (+18.3 to +31.2) | +19.9 (+12.6 to +27.2) | +14.8 (+3.3 to +26.5) |
| `rule_sweep6_handover_estimate`, scripted | `rule_sweep6_plume_border_estimate`, scripted | -59.9 (-65.0 to -54.8) | -57.1 (-64.7 to -50.1) | -61.0 (-69.1 to -53.0) | -68.1 (-81.4 to -54.8) |

### Choices of the meta agent

| Survey | Actions | Search | Plume | Border | Search, early third | Search, middle third | Search, late third | Switches per 100 decisions |
|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `lawnmower_sweep3_estimate` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `lawnmower_sweep4_path_end` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `minimal_values` | deterministic | 0.67 | 0.18 | 0.16 | 0.45 | 0.66 | 0.89 | 5.3 |
| `minimal_values` | sampled | 0.74 | 0.13 | 0.13 | 0.45 | 0.77 | 0.99 | 16.4 |
| `random_sweep4_handover_visit` | scripted | 0.33 | 0.34 | 0.33 | 0.33 | 0.33 | 0.33 | 66.4 |
| `random_sweep6_handover_estimate` | scripted | 0.33 | 0.34 | 0.33 | 0.33 | 0.33 | 0.33 | 66.4 |
| `random_trained_visit` | scripted | 0.33 | 0.34 | 0.33 | 0.33 | 0.33 | 0.33 | 66.4 |
| `rule_space_handover_estimate` | scripted | 0.78 | 0.06 | 0.16 | 0.53 | 0.84 | 0.98 | 9.9 |
| `rule_sweep4_handover_visit` | scripted | 0.69 | 0.11 | 0.19 | 0.65 | 0.62 | 0.81 | 15.9 |
| `rule_sweep4_plume_border_visit` | scripted | 0.54 | 0.17 | 0.29 | 0.60 | 0.43 | 0.60 | 24.9 |
| `rule_sweep4_scripted_gas_visit` | scripted | 0.75 | 0.07 | 0.18 | 0.68 | 0.69 | 0.88 | 7.5 |
| `rule_sweep6_handover_estimate` | scripted | 0.67 | 0.10 | 0.23 | 0.62 | 0.57 | 0.82 | 15.1 |
| `rule_sweep6_plume_border_estimate` | scripted | 0.54 | 0.15 | 0.31 | 0.55 | 0.41 | 0.66 | 23.7 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 0.68 | 0.10 | 0.22 | 0.62 | 0.59 | 0.83 | 10.6 |
| `rule_trained_visit` | scripted | 0.60 | 0.15 | 0.25 | 0.44 | 0.56 | 0.80 | 20.8 |
| `space_est` | deterministic | 0.69 | 0.07 | 0.24 | 0.50 | 0.74 | 0.84 | 3.4 ± 0.2 |
| `space_est` | sampled | 0.76 | 0.08 | 0.16 | 0.52 | 0.81 | 0.97 | 11.3 ± 0.8 |
| `space_only_visit` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `sweep4_visit` | deterministic | 0.71 | 0.11 | 0.19 | 0.66 | 0.65 | 0.81 | 7.1 ± 0.0 |
| `sweep4_visit` | sampled | 0.72 | 0.10 | 0.18 | 0.67 | 0.66 | 0.83 | 8.7 ± 0.3 |
| `sweep6_est` | deterministic | 0.69 | 0.09 | 0.22 | 0.64 | 0.62 | 0.82 | 8.4 ± 0.2 |
| `sweep6_est` | sampled | 0.69 | 0.09 | 0.22 | 0.64 | 0.62 | 0.82 | 9.6 ± 0.3 |
| `sweep6_est_control` | deterministic | 0.57 | 0.08 | 0.34 | 0.58 | 0.44 | 0.70 | 9.5 ± 0.3 |
| `sweep6_est_control` | sampled | 0.65 | 0.13 | 0.22 | 0.61 | 0.55 | 0.81 | 22.9 ± 0.4 |
| `sweep6_est_map` | deterministic | 0.72 | 0.23 | 0.04 | 0.92 | 0.73 | 0.51 | 6.3 ± 0.4 |
| `sweep6_est_map` | sampled | 0.72 | 0.21 | 0.07 | 0.93 | 0.71 | 0.53 | 11.3 ± 1.6 |
| `sweep6_est_no_time` | deterministic | 0.69 | 0.09 | 0.23 | 0.64 | 0.61 | 0.82 | 8.6 ± 0.3 |
| `sweep6_est_no_time` | sampled | 0.69 | 0.09 | 0.23 | 0.64 | 0.61 | 0.82 | 9.7 ± 0.3 |
| `sweep6_est_one` | deterministic | 0.69 | 0.07 | 0.24 | 0.64 | 0.61 | 0.82 | 8.5 ± 0.5 |
| `sweep6_est_one` | sampled | 0.69 | 0.07 | 0.24 | 0.64 | 0.61 | 0.81 | 9.8 ± 0.7 |
| `untuned` | deterministic | 0.34 | 0.41 | 0.25 | 0.33 | 0.32 | 0.36 | 14.8 |
| `untuned` | sampled | 0.61 | 0.19 | 0.20 | 0.36 | 0.60 | 0.88 | 29.9 |

## one_source: sources [1, 1], plume width [2.0, 5.0]

### Surveys

| Survey | Actions | Seeds | Ended by goal (%) | AUV steps | Overlap plume | Overlap border | Visited plume (%) | Visited border (%) | Plumes found (%) | Turns |
|---|---|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1 | 100.0 | 366.5 | 0.996 | 0.991 | 99.5 | 98.2 | 99.8 | 34.6 |
| `lawnmower_sweep3_estimate` | scripted | 1 | 99.5 | 282.9 | 0.993 | 0.982 | 85.1 | 85.1 | 100.0 | 26.0 |
| `minimal_values` | deterministic | 1 | 85.9 | 340.6 | 0.990 | 0.973 | 96.1 | 95.3 | 99.6 | 95.2 |
| `minimal_values` | sampled | 1 | 88.9 | 344.8 | 0.995 | 0.988 | 96.8 | 96.1 | 99.8 | 87.4 |
| `random_trained_visit` | scripted | 1 | 17.6 | 471.3 | 0.954 | 0.938 | 88.1 | 86.1 | 96.7 | 241.6 |
| `rule_sweep6_handover_estimate` | scripted | 1 | 100.0 | 197.1 | 0.997 | 0.990 | 94.1 | 95.0 | 100.0 | 40.5 |
| `rule_trained_visit` | scripted | 1 | 85.4 | 344.5 | 0.992 | 0.979 | 94.7 | 94.2 | 99.7 | 100.4 |
| `sweep6_est` | deterministic | 3 | 99.8 ± 0.0 | 192.4 ± 0.3 | 0.997 ± 0.000 | 0.989 ± 0.001 | 94.3 ± 0.5 | 95.3 ± 0.1 | 100.0 ± 0.0 | 37.2 ± 0.6 |
| `sweep6_est` | sampled | 3 | 99.9 ± 0.0 | 192.7 ± 0.7 | 0.997 ± 0.000 | 0.990 ± 0.001 | 94.4 ± 0.6 | 95.4 ± 0.2 | 100.0 ± 0.0 | 37.6 ± 0.6 |
| `sweep6_est_one` | deterministic | 3 | 99.7 ± 0.1 | 193.5 ± 1.2 | 0.997 ± 0.000 | 0.989 ± 0.000 | 93.7 ± 0.3 | 95.8 ± 0.4 | 100.0 ± 0.0 | 37.9 ± 1.2 |
| `sweep6_est_one` | sampled | 3 | 99.9 ± 0.0 | 194.1 ± 1.3 | 0.997 ± 0.000 | 0.990 ± 0.000 | 93.8 ± 0.3 | 95.9 ± 0.4 | 100.0 ± 0.0 | 38.1 ± 1.1 |
| `untuned` | deterministic | 1 | 51.3 | 403.6 | 0.980 | 0.964 | 94.3 | 92.5 | 98.9 | 194.1 |
| `untuned` | sampled | 1 | 86.4 | 376.7 | 0.992 | 0.985 | 97.0 | 95.6 | 99.6 | 113.8 |

### Steps and plumes found per number of true plumes

| Survey | Actions | Steps, 1 plume(s) | Found (%), 1 plume(s) | Steps, 2 plume(s) | Found (%), 2 plume(s) | Steps, 3 plume(s) | Found (%), 3 plume(s) |
|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 366 | 99.8 | - | - | - | - |
| `lawnmower_sweep3_estimate` | scripted | 283 | 100.0 | - | - | - | - |
| `minimal_values` | deterministic | 341 | 99.6 | - | - | - | - |
| `minimal_values` | sampled | 345 | 99.8 | - | - | - | - |
| `random_trained_visit` | scripted | 471 | 96.7 | - | - | - | - |
| `rule_sweep6_handover_estimate` | scripted | 197 | 100.0 | - | - | - | - |
| `rule_trained_visit` | scripted | 344 | 99.7 | - | - | - | - |
| `sweep6_est` | deterministic | 192 ± 0 | 100.0 ± 0.0 | - | - | - | - |
| `sweep6_est` | sampled | 193 ± 1 | 100.0 ± 0.0 | - | - | - | - |
| `sweep6_est_one` | deterministic | 194 ± 1 | 100.0 ± 0.0 | - | - | - | - |
| `sweep6_est_one` | sampled | 194 ± 1 | 100.0 ± 0.0 | - | - | - | - |
| `untuned` | deterministic | 404 | 98.9 | - | - | - | - |
| `untuned` | sampled | 377 | 99.6 | - | - | - | - |

### Each seed against the lawnmower and the rule

| Variant | Actions | Seed | Against | Difference in AUV steps (95 % interval) | Ended (%) | Overlap plume / border | Plumes found (%) | Beats it |
|---|---|---|---|---|---|---|---|---|
| `sweep6_est` | deterministic | 1 | `lawnmower_sweep3_estimate` | -90.8 (-93.2 to -88.6) | 99.8 | 0.997 / 0.988 | 100.0 | yes |
| `sweep6_est` | deterministic | 1 | `rule_sweep6_handover_estimate` | -5.1 (-6.5 to -3.5) | 99.8 | 0.997 / 0.988 | 100.0 | no |
| `sweep6_est` | deterministic | 2 | `lawnmower_sweep3_estimate` | -90.2 (-92.6 to -88.0) | 99.8 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est` | deterministic | 2 | `rule_sweep6_handover_estimate` | -4.5 (-5.9 to -3.0) | 99.8 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est` | deterministic | 3 | `lawnmower_sweep3_estimate` | -90.3 (-92.5 to -88.1) | 99.8 | 0.998 / 0.990 | 100.0 | yes |
| `sweep6_est` | deterministic | 3 | `rule_sweep6_handover_estimate` | -4.6 (-5.8 to -3.2) | 99.8 | 0.998 / 0.990 | 100.0 | yes |
| `sweep6_est` | sampled | 1 | `lawnmower_sweep3_estimate` | -91.2 (-93.5 to -89.1) | 99.9 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est` | sampled | 1 | `rule_sweep6_handover_estimate` | -5.5 (-6.6 to -4.1) | 99.9 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est` | sampled | 2 | `lawnmower_sweep3_estimate` | -89.7 (-92.0 to -87.5) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est` | sampled | 2 | `rule_sweep6_handover_estimate` | -4.0 (-5.3 to -2.7) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est` | sampled | 3 | `lawnmower_sweep3_estimate` | -89.6 (-91.8 to -87.5) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est` | sampled | 3 | `rule_sweep6_handover_estimate` | -3.8 (-4.9 to -2.6) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 1 | `lawnmower_sweep3_estimate` | -89.5 (-91.9 to -87.1) | 99.8 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 1 | `rule_sweep6_handover_estimate` | -3.8 (-5.1 to -2.3) | 99.8 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est_one` | deterministic | 2 | `lawnmower_sweep3_estimate` | -90.7 (-93.1 to -88.4) | 99.8 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 2 | `rule_sweep6_handover_estimate` | -5.0 (-6.3 to -3.4) | 99.8 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est_one` | deterministic | 3 | `lawnmower_sweep3_estimate` | -87.7 (-90.2 to -85.3) | 99.6 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 3 | `rule_sweep6_handover_estimate` | -2.0 (-3.4 to -0.4) | 99.6 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est_one` | sampled | 1 | `lawnmower_sweep3_estimate` | -88.3 (-90.7 to -85.9) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | sampled | 1 | `rule_sweep6_handover_estimate` | -2.6 (-4.0 to -1.0) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | sampled | 2 | `lawnmower_sweep3_estimate` | -90.5 (-92.8 to -88.3) | 100.0 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | sampled | 2 | `rule_sweep6_handover_estimate` | -4.8 (-6.0 to -3.6) | 100.0 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | sampled | 3 | `lawnmower_sweep3_estimate` | -87.5 (-89.7 to -85.1) | 99.9 | 0.997 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | sampled | 3 | `rule_sweep6_handover_estimate` | -1.7 (-2.9 to -0.5) | 99.9 | 0.997 / 0.990 | 100.0 | yes |

### Paired differences in AUV steps, each side the mean over its seeds

| Survey | Against | All episodes | 1 plume | 2 plumes | 3 plumes |
|---|---|---|---|---|---|
| `sweep6_est`, sampled | `rule_sweep6_handover_estimate`, scripted | -4.4 (-5.4 to -3.3) | -4.4 (-5.4 to -3.3) | - | - |
| `sweep6_est`, sampled | `lawnmower_sweep3_estimate`, scripted | -90.2 (-92.4 to -88.1) | -90.2 (-92.4 to -88.1) | - | - |
| `sweep6_est`, sampled | `sweep6_est_one`, sampled | -1.4 (-2.1 to -0.8) | -1.4 (-2.1 to -0.8) | - | - |
| `sweep6_est`, deterministic | `sweep6_est`, sampled | -0.3 (-1.0 to +0.4) | -0.3 (-1.0 to +0.4) | - | - |

### Choices of the meta agent

| Survey | Actions | Search | Plume | Border | Search, early third | Search, middle third | Search, late third | Switches per 100 decisions |
|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `lawnmower_sweep3_estimate` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `minimal_values` | deterministic | 0.79 | 0.11 | 0.10 | 0.61 | 0.83 | 0.94 | 3.6 |
| `minimal_values` | sampled | 0.83 | 0.08 | 0.09 | 0.61 | 0.90 | 0.99 | 11.0 |
| `random_trained_visit` | scripted | 0.33 | 0.34 | 0.33 | 0.33 | 0.33 | 0.33 | 66.4 |
| `rule_sweep6_handover_estimate` | scripted | 0.78 | 0.06 | 0.15 | 0.81 | 0.67 | 0.88 | 10.8 |
| `rule_trained_visit` | scripted | 0.73 | 0.10 | 0.17 | 0.59 | 0.73 | 0.87 | 14.2 |
| `sweep6_est` | deterministic | 0.81 | 0.05 | 0.14 | 0.82 | 0.72 | 0.89 | 5.7 ± 0.0 |
| `sweep6_est` | sampled | 0.81 | 0.05 | 0.14 | 0.82 | 0.71 | 0.89 | 6.4 ± 0.1 |
| `sweep6_est_one` | deterministic | 0.80 | 0.04 | 0.15 | 0.82 | 0.71 | 0.89 | 5.8 ± 0.3 |
| `sweep6_est_one` | sampled | 0.80 | 0.04 | 0.15 | 0.82 | 0.70 | 0.89 | 6.6 ± 0.4 |
| `untuned` | deterministic | 0.48 | 0.34 | 0.18 | 0.47 | 0.47 | 0.50 | 11.6 |
| `untuned` | sampled | 0.74 | 0.12 | 0.14 | 0.51 | 0.76 | 0.95 | 22.5 |

## small_plumes: sources [1, 3], plume width [1.0, 2.0]

### Surveys

| Survey | Actions | Seeds | Ended by goal (%) | AUV steps | Overlap plume | Overlap border | Visited plume (%) | Visited border (%) | Plumes found (%) | Turns |
|---|---|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1 | 100.0 | 369.4 | 0.882 | 0.777 | 96.5 | 96.1 | 96.7 | 34.6 |
| `lawnmower_sweep3_estimate` | scripted | 1 | 98.8 | 285.9 | 0.784 | 0.667 | 84.6 | 85.0 | 95.2 | 26.9 |
| `minimal_values` | deterministic | 1 | 47.0 | 428.8 | 0.687 | 0.562 | 85.9 | 84.9 | 86.5 | 140.4 |
| `minimal_values` | sampled | 1 | 52.4 | 425.5 | 0.767 | 0.636 | 92.0 | 91.1 | 94.8 | 118.9 |
| `rule_sweep6_handover_estimate` | scripted | 1 | 99.2 | 275.7 | 0.845 | 0.741 | 92.6 | 92.0 | 92.1 | 81.5 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 1 | 100.0 | 248.9 | 0.829 | 0.714 | 90.2 | 90.5 | 91.0 | 53.9 |
| `sweep6_est` | deterministic | 3 | 96.0 ± 0.6 | 284.5 ± 0.8 | 0.829 ± 0.005 | 0.724 ± 0.003 | 92.0 ± 0.1 | 91.8 ± 0.2 | 92.7 ± 0.3 | 84.7 ± 1.3 |
| `sweep6_est` | sampled | 3 | 97.2 ± 0.6 | 282.5 ± 1.6 | 0.834 ± 0.003 | 0.730 ± 0.003 | 92.1 ± 0.1 | 91.9 ± 0.1 | 92.5 ± 0.1 | 83.4 ± 1.5 |
| `sweep6_est_map` | deterministic | 3 | 92.7 ± 3.2 | 308.2 ± 7.3 | 0.796 ± 0.002 | 0.659 ± 0.004 | 89.2 ± 0.5 | 88.5 ± 0.3 | 89.4 ± 0.1 | 93.1 ± 15.0 |
| `sweep6_est_map` | sampled | 3 | 97.5 ± 0.2 | 294.4 ± 4.5 | 0.824 ± 0.013 | 0.688 ± 0.016 | 90.8 ± 1.7 | 90.0 ± 1.3 | 90.5 ± 1.3 | 78.6 ± 4.3 |

### Steps and plumes found per number of true plumes

| Survey | Actions | Steps, 1 plume(s) | Found (%), 1 plume(s) | Steps, 2 plume(s) | Found (%), 2 plume(s) | Steps, 3 plume(s) | Found (%), 3 plume(s) |
|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 365 | 96.3 | 370 | 96.6 | 375 | 97.0 |
| `lawnmower_sweep3_estimate` | scripted | 286 | 96.3 | 285 | 93.9 | 287 | 95.9 |
| `minimal_values` | deterministic | 400 | 95.7 | 441 | 87.6 | 454 | 80.8 |
| `minimal_values` | sampled | 401 | 94.1 | 438 | 95.5 | 444 | 94.3 |
| `rule_sweep6_handover_estimate` | scripted | 231 | 93.6 | 290 | 91.0 | 321 | 92.4 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 215 | 92.6 | 260 | 88.4 | 285 | 93.0 |
| `sweep6_est` | deterministic | 237 ± 2 | 93.6 ± 0.0 | 298 ± 2 | 90.3 ± 0.3 | 337 ± 1 | 94.7 ± 0.3 |
| `sweep6_est` | sampled | 236 ± 2 | 93.6 ± 0.0 | 295 ± 2 | 90.1 ± 0.3 | 334 ± 1 | 94.3 ± 0.0 |
| `sweep6_est_map` | deterministic | 272 ± 7 | 90.2 ± 2.8 | 320 ± 5 | 87.3 ± 0.2 | 346 ± 12 | 91.1 ± 1.4 |
| `sweep6_est_map` | sampled | 254 ± 6 | 91.3 ± 2.5 | 309 ± 4 | 88.2 ± 1.7 | 334 ± 7 | 92.5 ± 0.5 |

### Each seed against the lawnmower and the rule

| Variant | Actions | Seed | Against | Difference in AUV steps (95 % interval) | Ended (%) | Overlap plume / border | Plumes found (%) | Beats it |
|---|---|---|---|---|---|---|---|---|
| `sweep6_est` | deterministic | 1 | `lawnmower_sweep3_estimate` | -1.9 (-8.7 to +5.1) | 96.8 | 0.834 / 0.727 | 93.0 | no |
| `sweep6_est` | deterministic | 1 | `rule_sweep6_handover_estimate` | +8.3 (+3.4 to +13.6) | 96.8 | 0.834 / 0.727 | 93.0 | no |
| `sweep6_est` | deterministic | 2 | `lawnmower_sweep3_estimate` | -0.3 (-7.4 to +6.9) | 95.6 | 0.832 / 0.725 | 92.6 | no |
| `sweep6_est` | deterministic | 2 | `rule_sweep6_handover_estimate` | +9.9 (+4.6 to +15.3) | 95.6 | 0.832 / 0.725 | 92.6 | no |
| `sweep6_est` | deterministic | 3 | `lawnmower_sweep3_estimate` | -2.1 (-9.1 to +4.7) | 95.6 | 0.823 / 0.721 | 92.4 | no |
| `sweep6_est` | deterministic | 3 | `rule_sweep6_handover_estimate` | +8.1 (+2.9 to +13.4) | 95.6 | 0.823 / 0.721 | 92.4 | no |
| `sweep6_est` | sampled | 1 | `lawnmower_sweep3_estimate` | -5.7 (-12.0 to +0.9) | 97.8 | 0.834 / 0.729 | 92.3 | no |
| `sweep6_est` | sampled | 1 | `rule_sweep6_handover_estimate` | +4.5 (+0.1 to +9.2) | 97.8 | 0.834 / 0.729 | 92.3 | no |
| `sweep6_est` | sampled | 2 | `lawnmower_sweep3_estimate` | -2.5 (-9.2 to +4.4) | 97.4 | 0.837 / 0.735 | 92.5 | no |
| `sweep6_est` | sampled | 2 | `rule_sweep6_handover_estimate` | +7.7 (+2.8 to +12.5) | 97.4 | 0.837 / 0.735 | 92.5 | no |
| `sweep6_est` | sampled | 3 | `lawnmower_sweep3_estimate` | -2.0 (-8.7 to +4.5) | 96.4 | 0.830 / 0.727 | 92.6 | no |
| `sweep6_est` | sampled | 3 | `rule_sweep6_handover_estimate` | +8.2 (+3.0 to +13.9) | 96.4 | 0.830 / 0.727 | 92.6 | no |
| `sweep6_est_map` | deterministic | 1 | `lawnmower_sweep3_estimate` | +14.1 (+6.2 to +21.5) | 95.6 | 0.799 / 0.653 | 89.3 | no |
| `sweep6_est_map` | deterministic | 1 | `rule_sweep6_handover_estimate` | +24.3 (+17.4 to +31.1) | 95.6 | 0.799 / 0.653 | 89.3 | no |
| `sweep6_est_map` | deterministic | 2 | `lawnmower_sweep3_estimate` | +31.8 (+23.6 to +40.1) | 94.4 | 0.795 / 0.660 | 89.5 | no |
| `sweep6_est_map` | deterministic | 2 | `rule_sweep6_handover_estimate` | +42.0 (+34.9 to +49.3) | 94.4 | 0.795 / 0.660 | 89.5 | no |
| `sweep6_est_map` | deterministic | 3 | `lawnmower_sweep3_estimate` | +21.1 (+12.7 to +29.4) | 88.2 | 0.794 / 0.663 | 89.4 | no |
| `sweep6_est_map` | deterministic | 3 | `rule_sweep6_handover_estimate` | +31.3 (+24.3 to +38.6) | 88.2 | 0.794 / 0.663 | 89.4 | no |
| `sweep6_est_map` | sampled | 1 | `lawnmower_sweep3_estimate` | +6.5 (-0.7 to +13.7) | 97.8 | 0.811 / 0.675 | 89.6 | no |
| `sweep6_est_map` | sampled | 1 | `rule_sweep6_handover_estimate` | +16.7 (+10.4 to +23.4) | 97.8 | 0.811 / 0.675 | 89.6 | no |
| `sweep6_est_map` | sampled | 2 | `lawnmower_sweep3_estimate` | +14.8 (+7.2 to +22.0) | 97.2 | 0.820 / 0.680 | 89.6 | no |
| `sweep6_est_map` | sampled | 2 | `rule_sweep6_handover_estimate` | +25.0 (+18.5 to +31.5) | 97.2 | 0.820 / 0.680 | 89.6 | no |
| `sweep6_est_map` | sampled | 3 | `lawnmower_sweep3_estimate` | +4.2 (-2.6 to +10.9) | 97.6 | 0.841 / 0.710 | 92.3 | no |
| `sweep6_est_map` | sampled | 3 | `rule_sweep6_handover_estimate` | +14.4 (+8.8 to +20.1) | 97.6 | 0.841 / 0.710 | 92.3 | no |

### Paired differences in AUV steps, each side the mean over its seeds

| Survey | Against | All episodes | 1 plume | 2 plumes | 3 plumes |
|---|---|---|---|---|---|
| `sweep6_est`, sampled | `rule_sweep6_handover_estimate`, scripted | +6.8 (+2.5 to +11.1) | +4.9 (-1.2 to +10.9) | +4.7 (-2.8 to +12.0) | +12.8 (+3.0 to +22.4) |
| `sweep6_est`, sampled | `rule_sweep6_scripted_gas_estimate`, scripted | +33.6 (+29.0 to +38.1) | +21.5 (+14.9 to +27.9) | +35.4 (+27.7 to +43.0) | +49.3 (+40.5 to +58.3) |
| `sweep6_est`, sampled | `lawnmower_sweep3_estimate`, scripted | -3.4 (-9.7 to +2.7) | -49.4 (-59.0 to -40.7) | +9.5 (+0.8 to +18.3) | +46.9 (+37.2 to +56.5) |
| `sweep6_est`, sampled | `sweep6_est_map`, sampled | -11.9 (-16.7 to -7.1) | -18.3 (-25.3 to -11.0) | -13.6 (-21.8 to -4.7) | +0.4 (-8.5 to +9.2) |
| `sweep6_est`, deterministic | `sweep6_est`, sampled | +2.0 (+0.2 to +3.8) | +1.0 (-1.6 to +3.5) | +2.9 (-0.0 to +6.0) | +2.3 (-1.5 to +6.1) |

### Choices of the meta agent

| Survey | Actions | Search | Plume | Border | Search, early third | Search, middle third | Search, late third | Switches per 100 decisions |
|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `lawnmower_sweep3_estimate` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `minimal_values` | deterministic | 0.67 | 0.03 | 0.30 | 0.56 | 0.63 | 0.81 | 6.0 |
| `minimal_values` | sampled | 0.75 | 0.05 | 0.20 | 0.59 | 0.70 | 0.95 | 19.2 |
| `rule_sweep6_handover_estimate` | scripted | 0.59 | 0.09 | 0.32 | 0.67 | 0.47 | 0.62 | 20.7 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 0.66 | 0.07 | 0.28 | 0.71 | 0.54 | 0.71 | 18.2 |
| `sweep6_est` | deterministic | 0.57 | 0.05 | 0.38 | 0.67 | 0.45 | 0.60 | 15.1 ± 0.1 |
| `sweep6_est` | sampled | 0.58 | 0.04 | 0.38 | 0.68 | 0.45 | 0.61 | 15.4 ± 0.1 |
| `sweep6_est_map` | deterministic | 0.65 | 0.29 | 0.07 | 0.90 | 0.54 | 0.50 | 10.0 ± 0.9 |
| `sweep6_est_map` | sampled | 0.66 | 0.25 | 0.09 | 0.91 | 0.56 | 0.51 | 16.0 ± 3.2 |

## large_plumes: sources [1, 3], plume width [5.0, 8.0]

### Surveys

| Survey | Actions | Seeds | Ended by goal (%) | AUV steps | Overlap plume | Overlap border | Visited plume (%) | Visited border (%) | Plumes found (%) | Turns |
|---|---|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1 | 100.0 | 376.3 | 0.997 | 0.989 | 96.8 | 94.9 | 100.0 | 35.2 |
| `lawnmower_sweep3_estimate` | scripted | 1 | 100.0 | 285.6 | 0.998 | 0.992 | 83.8 | 83.7 | 100.0 | 25.6 |
| `minimal_values` | deterministic | 1 | 19.0 | 460.5 | 0.871 | 0.769 | 60.2 | 77.9 | 96.7 | 289.0 |
| `minimal_values` | sampled | 1 | 41.2 | 446.2 | 0.982 | 0.966 | 85.8 | 89.8 | 99.5 | 184.0 |
| `rule_sweep6_handover_estimate` | scripted | 1 | 93.8 | 338.6 | 1.000 | 0.997 | 91.1 | 92.5 | 100.0 | 105.3 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 1 | 99.4 | 301.1 | 0.999 | 0.998 | 90.9 | 92.3 | 100.0 | 43.2 |
| `sweep6_est` | deterministic | 3 | 97.2 ± 1.0 | 324.2 ± 1.5 | 0.995 ± 0.002 | 0.990 ± 0.003 | 90.9 ± 0.2 | 91.7 ± 0.2 | 99.8 ± 0.1 | 103.6 ± 1.9 |
| `sweep6_est` | sampled | 3 | 98.1 ± 0.2 | 324.8 ± 0.4 | 0.997 ± 0.001 | 0.994 ± 0.001 | 91.4 ± 0.3 | 92.2 ± 0.2 | 99.9 ± 0.1 | 101.7 ± 0.7 |
| `sweep6_est_map` | deterministic | 3 | 92.1 ± 1.6 | 334.3 ± 7.0 | 0.999 ± 0.001 | 0.996 ± 0.001 | 92.1 ± 2.1 | 82.4 ± 2.2 | 99.9 ± 0.1 | 86.6 ± 5.8 |
| `sweep6_est_map` | sampled | 3 | 95.0 ± 3.0 | 330.4 ± 8.2 | 0.999 ± 0.000 | 0.997 ± 0.000 | 91.2 ± 1.3 | 84.3 ± 1.6 | 100.0 ± 0.0 | 87.3 ± 8.4 |

### Steps and plumes found per number of true plumes

| Survey | Actions | Steps, 1 plume(s) | Found (%), 1 plume(s) | Steps, 2 plume(s) | Found (%), 2 plume(s) | Steps, 3 plume(s) | Found (%), 3 plume(s) |
|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 376 | 100.0 | 378 | 100.0 | 386 | 100.0 |
| `lawnmower_sweep3_estimate` | scripted | 286 | 100.0 | 283 | 100.0 | 290 | 100.0 |
| `minimal_values` | deterministic | 459 | 100.0 | 468 | 88.5 | 480 | 66.7 |
| `minimal_values` | sampled | 443 | 100.0 | 460 | 98.1 | 480 | 100.0 |
| `rule_sweep6_handover_estimate` | scripted | 333 | 100.0 | 369 | 100.0 | 463 | 100.0 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 298 | 100.0 | 319 | 100.0 | 376 | 100.0 |
| `sweep6_est` | deterministic | 318 ± 1 | 100.0 ± 0.0 | 357 ± 2 | 99.1 ± 0.3 | 399 ± 5 | 100.0 ± 0.0 |
| `sweep6_est` | sampled | 319 ± 0 | 100.0 ± 0.0 | 357 ± 4 | 99.6 ± 0.3 | 410 ± 7 | 100.0 ± 0.0 |
| `sweep6_est_map` | deterministic | 330 ± 7 | 99.9 ± 0.1 | 358 ± 8 | 100.0 ± 0.0 | 446 ± 32 | 100.0 ± 0.0 |
| `sweep6_est_map` | sampled | 324 ± 8 | 100.0 ± 0.0 | 361 ± 11 | 100.0 ± 0.0 | 444 ± 29 | 100.0 ± 0.0 |

### Each seed against the lawnmower and the rule

| Variant | Actions | Seed | Against | Difference in AUV steps (95 % interval) | Ended (%) | Overlap plume / border | Plumes found (%) | Beats it |
|---|---|---|---|---|---|---|---|---|
| `sweep6_est` | deterministic | 1 | `lawnmower_sweep3_estimate` | +36.8 (+29.7 to +43.6) | 97.2 | 0.996 / 0.992 | 99.8 | no |
| `sweep6_est` | deterministic | 1 | `rule_sweep6_handover_estimate` | -16.3 (-20.7 to -11.8) | 97.2 | 0.996 / 0.992 | 99.8 | no |
| `sweep6_est` | deterministic | 2 | `lawnmower_sweep3_estimate` | +38.5 (+31.4 to +45.3) | 98.4 | 0.996 / 0.992 | 99.8 | no |
| `sweep6_est` | deterministic | 2 | `rule_sweep6_handover_estimate` | -14.5 (-18.8 to -10.4) | 98.4 | 0.996 / 0.992 | 99.8 | no |
| `sweep6_est` | deterministic | 3 | `lawnmower_sweep3_estimate` | +40.5 (+33.5 to +47.4) | 96.0 | 0.992 / 0.986 | 99.7 | no |
| `sweep6_est` | deterministic | 3 | `rule_sweep6_handover_estimate` | -12.6 (-17.2 to -7.9) | 96.0 | 0.992 / 0.986 | 99.7 | no |
| `sweep6_est` | sampled | 1 | `lawnmower_sweep3_estimate` | +38.7 (+31.7 to +45.7) | 98.0 | 0.997 / 0.995 | 99.8 | no |
| `sweep6_est` | sampled | 1 | `rule_sweep6_handover_estimate` | -14.3 (-18.3 to -10.2) | 98.0 | 0.997 / 0.995 | 99.8 | no |
| `sweep6_est` | sampled | 2 | `lawnmower_sweep3_estimate` | +39.3 (+32.4 to +46.0) | 98.4 | 0.998 / 0.995 | 100.0 | no |
| `sweep6_est` | sampled | 2 | `rule_sweep6_handover_estimate` | -13.7 (-18.0 to -9.7) | 98.4 | 0.998 / 0.995 | 100.0 | no |
| `sweep6_est` | sampled | 3 | `lawnmower_sweep3_estimate` | +39.6 (+32.5 to +46.2) | 97.8 | 0.996 / 0.993 | 99.8 | no |
| `sweep6_est` | sampled | 3 | `rule_sweep6_handover_estimate` | -13.4 (-17.6 to -9.4) | 97.8 | 0.996 / 0.993 | 99.8 | no |
| `sweep6_est_map` | deterministic | 1 | `lawnmower_sweep3_estimate` | +38.8 (+32.5 to +45.0) | 92.4 | 0.999 / 0.997 | 100.0 | no |
| `sweep6_est_map` | deterministic | 1 | `rule_sweep6_handover_estimate` | -14.2 (-19.7 to -8.4) | 92.4 | 0.999 / 0.997 | 100.0 | no |
| `sweep6_est_map` | deterministic | 2 | `lawnmower_sweep3_estimate` | +54.0 (+47.4 to +60.8) | 94.0 | 0.999 / 0.996 | 100.0 | no |
| `sweep6_est_map` | deterministic | 2 | `rule_sweep6_handover_estimate` | +1.0 (-4.4 to +6.2) | 94.0 | 0.999 / 0.996 | 100.0 | no |
| `sweep6_est_map` | deterministic | 3 | `lawnmower_sweep3_estimate` | +53.3 (+45.5 to +60.9) | 90.0 | 0.997 / 0.995 | 99.8 | no |
| `sweep6_est_map` | deterministic | 3 | `rule_sweep6_handover_estimate` | +0.2 (-4.4 to +5.0) | 90.0 | 0.997 / 0.995 | 99.8 | no |
| `sweep6_est_map` | sampled | 1 | `lawnmower_sweep3_estimate` | +35.2 (+29.2 to +41.2) | 98.4 | 0.999 / 0.997 | 100.0 | no |
| `sweep6_est_map` | sampled | 1 | `rule_sweep6_handover_estimate` | -17.9 (-22.7 to -13.0) | 98.4 | 0.999 / 0.997 | 100.0 | no |
| `sweep6_est_map` | sampled | 2 | `lawnmower_sweep3_estimate` | +43.9 (+37.8 to +50.4) | 95.4 | 0.999 / 0.998 | 100.0 | no |
| `sweep6_est_map` | sampled | 2 | `rule_sweep6_handover_estimate` | -9.1 (-14.3 to -4.1) | 95.4 | 0.999 / 0.998 | 100.0 | no |
| `sweep6_est_map` | sampled | 3 | `lawnmower_sweep3_estimate` | +55.3 (+47.8 to +62.7) | 91.2 | 0.999 / 0.997 | 100.0 | no |
| `sweep6_est_map` | sampled | 3 | `rule_sweep6_handover_estimate` | +2.3 (-2.5 to +7.3) | 91.2 | 0.999 / 0.997 | 100.0 | no |

### Paired differences in AUV steps, each side the mean over its seeds

| Survey | Against | All episodes | 1 plume | 2 plumes | 3 plumes |
|---|---|---|---|---|---|
| `sweep6_est`, sampled | `rule_sweep6_handover_estimate`, scripted | -13.8 (-17.7 to -10.0) | -14.0 (-18.1 to -9.9) | -12.3 (-22.7 to -1.9) | -52.7 (-52.7 to -52.7) |
| `sweep6_est`, sampled | `rule_sweep6_scripted_gas_estimate`, scripted | +23.7 (+20.3 to +27.2) | +21.0 (+17.3 to +24.8) | +38.1 (+30.0 to +46.4) | +34.3 (+34.3 to +34.3) |
| `sweep6_est`, sampled | `lawnmower_sweep3_estimate`, scripted | +39.2 (+32.2 to +45.8) | +32.5 (+25.3 to +39.8) | +74.5 (+64.4 to +85.3) | +120.3 (+120.3 to +120.3) |
| `sweep6_est`, sampled | `sweep6_est_map`, sampled | -5.6 (-8.5 to -2.6) | -5.9 (-8.6 to -2.9) | -3.9 (-12.9 to +5.5) | -33.7 (-33.7 to -33.7) |
| `sweep6_est`, deterministic | `sweep6_est`, sampled | -0.6 (-2.3 to +1.2) | -0.6 (-2.3 to +1.3) | -0.4 (-5.0 to +5.1) | -11.7 (-11.7 to -11.7) |

### Choices of the meta agent

| Survey | Actions | Search | Plume | Border | Search, early third | Search, middle third | Search, late third | Switches per 100 decisions |
|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `lawnmower_sweep3_estimate` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `minimal_values` | deterministic | 0.22 | 0.06 | 0.72 | 0.14 | 0.17 | 0.34 | 3.2 |
| `minimal_values` | sampled | 0.48 | 0.17 | 0.36 | 0.16 | 0.41 | 0.87 | 23.6 |
| `rule_sweep6_handover_estimate` | scripted | 0.43 | 0.22 | 0.35 | 0.29 | 0.22 | 0.79 | 20.3 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 0.45 | 0.18 | 0.37 | 0.33 | 0.23 | 0.79 | 11.1 |
| `sweep6_est` | deterministic | 0.45 | 0.21 | 0.34 | 0.29 | 0.25 | 0.81 | 8.7 ± 0.4 |
| `sweep6_est` | sampled | 0.45 | 0.20 | 0.35 | 0.29 | 0.26 | 0.82 | 11.6 ± 0.8 |
| `sweep6_est_map` | deterministic | 0.51 | 0.37 | 0.12 | 0.79 | 0.40 | 0.35 | 11.0 ± 0.5 |
| `sweep6_est_map` | sampled | 0.51 | 0.35 | 0.14 | 0.80 | 0.37 | 0.37 | 19.4 ± 3.1 |
