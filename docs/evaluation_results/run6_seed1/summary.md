# Results of run 6, from summarise_evaluations.py

**Same episodes per scenario:** sources_1_3 yes

## sources_1_3: sources [1, 3], plume width [2.0, 5.0]

### Surveys

| Survey | Actions | Seeds | Ended by goal (%) | AUV steps | Overlap plume | Overlap border | Visited plume (%) | Visited border (%) | Plumes found (%) | Turns |
|---|---|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1 | 100.0 | 369.4 | 0.998 | 0.993 | 98.6 | 97.0 | 99.9 | 34.8 |
| `lawnmower_sweep3_estimate` | scripted | 1 | 99.6 | 283.7 | 0.993 | 0.983 | 84.4 | 84.0 | 100.0 | 25.9 |
| `rule_sweep4_handover_visit` | scripted | 1 | 99.0 | 303.0 | 0.998 | 0.993 | 97.0 | 95.9 | 100.0 | 77.2 |
| `rule_sweep6_handover_estimate` | scripted | 1 | 100.0 | 229.4 | 0.998 | 0.989 | 93.1 | 94.2 | 100.0 | 60.5 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 1 | 100.0 | 226.4 | 0.997 | 0.990 | 93.1 | 93.0 | 100.0 | 34.4 |
| `sweep4_visit` | deterministic | 3 | 97.4 ± 0.3 | 298.9 ± 0.5 | 0.993 ± 0.002 | 0.987 ± 0.002 | 96.7 ± 0.3 | 95.8 ± 0.2 | 99.8 ± 0.1 | 77.1 ± 0.5 |
| `sweep4_visit` | sampled | 3 | 99.4 ± 0.1 | 295.0 ± 0.3 | 0.999 ± 0.000 | 0.994 ± 0.000 | 97.3 ± 0.1 | 96.3 ± 0.0 | 100.0 ± 0.0 | 72.5 ± 0.4 |
| `sweep6_est` | deterministic | 3 | 99.5 ± 0.1 | 225.8 ± 1.1 | 0.996 ± 0.001 | 0.988 ± 0.000 | 93.5 ± 0.5 | 94.6 ± 0.1 | 99.9 ± 0.0 | 59.3 ± 0.9 |
| `sweep6_est` | sampled | 3 | 99.7 ± 0.1 | 226.1 ± 1.0 | 0.997 ± 0.000 | 0.989 ± 0.000 | 93.6 ± 0.4 | 94.7 ± 0.2 | 99.9 ± 0.1 | 59.2 ± 0.7 |
| `sweep6_est_control` | deterministic | 3 | 90.9 ± 1.5 | 264.6 ± 1.9 | 0.982 ± 0.001 | 0.965 ± 0.002 | 89.7 ± 0.1 | 92.2 ± 0.5 | 98.8 ± 0.2 | 97.5 ± 2.8 |
| `sweep6_est_control` | sampled | 3 | 99.8 ± 0.0 | 245.4 ± 0.9 | 0.997 ± 0.000 | 0.989 ± 0.001 | 95.1 ± 0.2 | 94.4 ± 0.1 | 100.0 ± 0.0 | 71.8 ± 0.8 |
| `sweep6_est_map` | deterministic | 3 | 99.3 ± 0.2 | 241.9 ± 5.2 | 0.993 ± 0.002 | 0.979 ± 0.002 | 92.8 ± 2.0 | 82.1 ± 1.1 | 99.9 ± 0.1 | 49.8 ± 1.4 |
| `sweep6_est_map` | sampled | 3 | 100.0 ± 0.0 | 242.8 ± 3.8 | 0.995 ± 0.000 | 0.982 ± 0.000 | 92.5 ± 1.4 | 83.5 ± 0.6 | 100.0 ± 0.0 | 50.3 ± 1.4 |
| `sweep6_est_no_time` | deterministic | 3 | 99.5 ± 0.1 | 227.0 ± 1.0 | 0.995 ± 0.001 | 0.987 ± 0.000 | 93.6 ± 0.1 | 94.8 ± 0.1 | 99.8 ± 0.0 | 60.3 ± 0.8 |
| `sweep6_est_no_time` | sampled | 3 | 99.7 ± 0.1 | 227.9 ± 0.8 | 0.997 ± 0.001 | 0.989 ± 0.001 | 93.7 ± 0.2 | 94.9 ± 0.1 | 99.9 ± 0.1 | 60.4 ± 0.7 |
| `sweep6_est_one` | deterministic | 3 | 99.3 ± 0.1 | 228.1 ± 2.9 | 0.995 ± 0.001 | 0.987 ± 0.000 | 92.7 ± 0.3 | 95.0 ± 0.4 | 99.8 ± 0.1 | 61.1 ± 1.9 |
| `sweep6_est_one` | sampled | 3 | 99.6 ± 0.0 | 228.3 ± 2.3 | 0.996 ± 0.000 | 0.989 ± 0.000 | 93.0 ± 0.4 | 95.2 ± 0.5 | 99.9 ± 0.0 | 61.0 ± 1.8 |

### Steps and plumes found per number of true plumes

| Survey | Actions | Steps, 1 plume(s) | Found (%), 1 plume(s) | Steps, 2 plume(s) | Found (%), 2 plume(s) | Steps, 3 plume(s) | Found (%), 3 plume(s) |
|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 367 | 100.0 | 371 | 99.9 | 372 | 100.0 |
| `lawnmower_sweep3_estimate` | scripted | 285 | 100.0 | 282 | 100.0 | 282 | 100.0 |
| `rule_sweep4_handover_visit` | scripted | 287 | 100.0 | 316 | 100.0 | 332 | 100.0 |
| `rule_sweep6_handover_estimate` | scripted | 213 | 100.0 | 241 | 100.0 | 262 | 100.0 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 208 | 100.0 | 240 | 100.0 | 259 | 100.0 |
| `sweep4_visit` | deterministic | 284 ± 1 | 99.9 ± 0.1 | 309 ± 0 | 99.7 ± 0.1 | 328 ± 1 | 99.7 ± 0.2 |
| `sweep4_visit` | sampled | 279 ± 1 | 100.0 ± 0.0 | 306 ± 1 | 100.0 ± 0.0 | 327 ± 1 | 100.0 ± 0.0 |
| `sweep6_est` | deterministic | 208 ± 1 | 100.0 ± 0.0 | 239 ± 1 | 99.7 ± 0.1 | 260 ± 2 | 100.0 ± 0.0 |
| `sweep6_est` | sampled | 208 ± 1 | 100.0 ± 0.0 | 239 ± 1 | 99.9 ± 0.1 | 261 ± 1 | 100.0 ± 0.0 |
| `sweep6_est_control` | deterministic | 247 ± 2 | 100.0 ± 0.0 | 279 ± 1 | 98.3 ± 0.2 | 292 ± 5 | 98.0 ± 0.4 |
| `sweep6_est_control` | sampled | 225 ± 1 | 100.0 ± 0.0 | 261 ± 1 | 100.0 ± 0.0 | 282 ± 3 | 100.0 ± 0.0 |
| `sweep6_est_map` | deterministic | 226 ± 4 | 99.9 ± 0.1 | 253 ± 8 | 99.9 ± 0.1 | 273 ± 6 | 99.8 ± 0.3 |
| `sweep6_est_map` | sampled | 225 ± 2 | 100.0 ± 0.0 | 255 ± 6 | 100.0 ± 0.0 | 281 ± 4 | 100.0 ± 0.0 |
| `sweep6_est_no_time` | deterministic | 209 ± 1 | 100.0 ± 0.0 | 241 ± 1 | 99.7 ± 0.1 | 260 ± 1 | 100.0 ± 0.0 |
| `sweep6_est_no_time` | sampled | 210 ± 0 | 100.0 ± 0.0 | 241 ± 1 | 99.8 ± 0.1 | 262 ± 1 | 100.0 ± 0.0 |
| `sweep6_est_one` | deterministic | 210 ± 2 | 100.0 ± 0.0 | 242 ± 4 | 99.6 ± 0.1 | 263 ± 4 | 99.9 ± 0.1 |
| `sweep6_est_one` | sampled | 210 ± 1 | 100.0 ± 0.0 | 241 ± 3 | 99.8 ± 0.1 | 264 ± 3 | 100.0 ± 0.0 |

### Each seed against the lawnmower and the rule

| Variant | Actions | Seed | Against | Difference in AUV steps (95 % interval) | Ended (%) | Overlap plume / border | Plumes found (%) | Beats it |
|---|---|---|---|---|---|---|---|---|
| `sweep4_visit` | deterministic | 1 | `lawnmower_sweep2_visit` | -70.8 (-73.6 to -67.7) | 97.6 | 0.995 / 0.989 | 99.8 | no |
| `sweep4_visit` | deterministic | 1 | `rule_sweep4_handover_visit` | -4.4 (-7.4 to -1.4) | 97.6 | 0.995 / 0.989 | 99.8 | no |
| `sweep4_visit` | deterministic | 2 | `lawnmower_sweep2_visit` | -71.0 (-73.8 to -68.0) | 97.6 | 0.994 / 0.987 | 99.8 | no |
| `sweep4_visit` | deterministic | 2 | `rule_sweep4_handover_visit` | -4.6 (-7.5 to -1.8) | 97.6 | 0.994 / 0.987 | 99.8 | no |
| `sweep4_visit` | deterministic | 3 | `lawnmower_sweep2_visit` | -69.8 (-72.7 to -66.7) | 96.9 | 0.991 / 0.984 | 99.6 | no |
| `sweep4_visit` | deterministic | 3 | `rule_sweep4_handover_visit` | -3.5 (-6.4 to -0.4) | 96.9 | 0.991 / 0.984 | 99.6 | no |
| `sweep4_visit` | sampled | 1 | `lawnmower_sweep2_visit` | -73.9 (-76.5 to -71.4) | 99.6 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 1 | `rule_sweep4_handover_visit` | -7.5 (-9.8 to -5.3) | 99.6 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 2 | `lawnmower_sweep2_visit` | -74.7 (-77.2 to -72.1) | 99.3 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 2 | `rule_sweep4_handover_visit` | -8.4 (-10.9 to -5.8) | 99.3 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 3 | `lawnmower_sweep2_visit` | -74.6 (-77.0 to -71.9) | 99.4 | 0.999 / 0.994 | 100.0 | yes |
| `sweep4_visit` | sampled | 3 | `rule_sweep4_handover_visit` | -8.2 (-10.5 to -5.9) | 99.4 | 0.999 / 0.994 | 100.0 | yes |
| `sweep6_est` | deterministic | 1 | `lawnmower_sweep3_estimate` | -59.3 (-62.5 to -56.1) | 99.5 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est` | deterministic | 1 | `rule_sweep6_handover_estimate` | -5.0 (-6.6 to -3.3) | 99.5 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est` | deterministic | 2 | `lawnmower_sweep3_estimate` | -57.3 (-60.4 to -54.1) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est` | deterministic | 2 | `rule_sweep6_handover_estimate` | -3.0 (-4.6 to -1.3) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est` | deterministic | 3 | `lawnmower_sweep3_estimate` | -56.9 (-60.1 to -53.7) | 99.4 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est` | deterministic | 3 | `rule_sweep6_handover_estimate` | -2.6 (-4.1 to -0.8) | 99.4 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est` | sampled | 1 | `lawnmower_sweep3_estimate` | -59.1 (-62.1 to -55.9) | 99.7 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est` | sampled | 1 | `rule_sweep6_handover_estimate` | -4.8 (-6.4 to -3.1) | 99.7 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est` | sampled | 2 | `lawnmower_sweep3_estimate` | -56.9 (-59.9 to -53.8) | 99.8 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est` | sampled | 2 | `rule_sweep6_handover_estimate` | -2.6 (-4.0 to -1.2) | 99.8 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est` | sampled | 3 | `lawnmower_sweep3_estimate` | -56.9 (-60.0 to -53.8) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est` | sampled | 3 | `rule_sweep6_handover_estimate` | -2.6 (-4.1 to -1.0) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_control` | deterministic | 1 | `lawnmower_sweep3_estimate` | -20.2 (-26.0 to -14.0) | 91.3 | 0.981 / 0.965 | 98.5 | no |
| `sweep6_est_control` | deterministic | 1 | `rule_sweep6_handover_estimate` | +34.1 (+29.4 to +38.9) | 91.3 | 0.981 / 0.965 | 98.5 | no |
| `sweep6_est_control` | deterministic | 2 | `lawnmower_sweep3_estimate` | -20.6 (-26.5 to -14.5) | 92.4 | 0.984 / 0.968 | 98.9 | no |
| `sweep6_est_control` | deterministic | 2 | `rule_sweep6_handover_estimate` | +33.7 (+28.7 to +38.6) | 92.4 | 0.984 / 0.968 | 98.9 | no |
| `sweep6_est_control` | deterministic | 3 | `lawnmower_sweep3_estimate` | -16.4 (-22.6 to -10.0) | 88.9 | 0.982 / 0.963 | 98.8 | no |
| `sweep6_est_control` | deterministic | 3 | `rule_sweep6_handover_estimate` | +37.9 (+32.6 to +43.3) | 88.9 | 0.982 / 0.963 | 98.8 | no |
| `sweep6_est_control` | sampled | 1 | `lawnmower_sweep3_estimate` | -38.7 (-42.3 to -35.1) | 99.8 | 0.998 / 0.990 | 100.0 | yes |
| `sweep6_est_control` | sampled | 1 | `rule_sweep6_handover_estimate` | +15.6 (+13.9 to +17.5) | 99.8 | 0.998 / 0.990 | 100.0 | no |
| `sweep6_est_control` | sampled | 2 | `lawnmower_sweep3_estimate` | -39.2 (-42.8 to -35.5) | 99.9 | 0.997 / 0.988 | 100.0 | yes |
| `sweep6_est_control` | sampled | 2 | `rule_sweep6_handover_estimate` | +15.1 (+13.4 to +16.9) | 99.9 | 0.997 / 0.988 | 100.0 | no |
| `sweep6_est_control` | sampled | 3 | `lawnmower_sweep3_estimate` | -37.1 (-40.8 to -33.4) | 99.8 | 0.997 / 0.989 | 100.0 | yes |
| `sweep6_est_control` | sampled | 3 | `rule_sweep6_handover_estimate` | +17.2 (+15.3 to +19.1) | 99.8 | 0.997 / 0.989 | 100.0 | no |
| `sweep6_est_map` | deterministic | 1 | `lawnmower_sweep3_estimate` | -37.2 (-40.3 to -34.3) | 99.0 | 0.994 / 0.978 | 100.0 | no |
| `sweep6_est_map` | deterministic | 1 | `rule_sweep6_handover_estimate` | +17.1 (+14.6 to +19.5) | 99.0 | 0.994 / 0.978 | 100.0 | no |
| `sweep6_est_map` | deterministic | 2 | `lawnmower_sweep3_estimate` | -39.0 (-42.2 to -35.7) | 99.6 | 0.996 / 0.982 | 100.0 | no |
| `sweep6_est_map` | deterministic | 2 | `rule_sweep6_handover_estimate` | +15.3 (+12.9 to +17.7) | 99.6 | 0.996 / 0.982 | 100.0 | no |
| `sweep6_est_map` | deterministic | 3 | `lawnmower_sweep3_estimate` | -49.1 (-52.4 to -45.6) | 99.2 | 0.990 / 0.976 | 99.7 | no |
| `sweep6_est_map` | deterministic | 3 | `rule_sweep6_handover_estimate` | +5.2 (+2.8 to +7.7) | 99.2 | 0.990 / 0.976 | 99.7 | no |
| `sweep6_est_map` | sampled | 1 | `lawnmower_sweep3_estimate` | -40.6 (-43.4 to -37.9) | 100.0 | 0.995 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 1 | `rule_sweep6_handover_estimate` | +13.7 (+11.8 to +15.5) | 100.0 | 0.995 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 2 | `lawnmower_sweep3_estimate` | -36.3 (-39.7 to -33.2) | 100.0 | 0.996 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 2 | `rule_sweep6_handover_estimate` | +18.0 (+15.7 to +20.1) | 100.0 | 0.996 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 3 | `lawnmower_sweep3_estimate` | -45.6 (-48.7 to -42.5) | 100.0 | 0.995 / 0.982 | 100.0 | no |
| `sweep6_est_map` | sampled | 3 | `rule_sweep6_handover_estimate` | +8.7 (+6.7 to +10.8) | 100.0 | 0.995 / 0.982 | 100.0 | no |
| `sweep6_est_no_time` | deterministic | 1 | `lawnmower_sweep3_estimate` | -57.5 (-60.5 to -54.4) | 99.4 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_no_time` | deterministic | 1 | `rule_sweep6_handover_estimate` | -3.2 (-4.7 to -1.4) | 99.4 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_no_time` | deterministic | 2 | `lawnmower_sweep3_estimate` | -55.2 (-58.3 to -51.9) | 99.4 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_no_time` | deterministic | 2 | `rule_sweep6_handover_estimate` | -0.9 (-2.5 to +0.9) | 99.4 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_no_time` | deterministic | 3 | `lawnmower_sweep3_estimate` | -57.3 (-60.5 to -54.0) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_no_time` | deterministic | 3 | `rule_sweep6_handover_estimate` | -3.0 (-4.5 to -1.3) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_no_time` | sampled | 1 | `lawnmower_sweep3_estimate` | -56.4 (-59.6 to -53.2) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_no_time` | sampled | 1 | `rule_sweep6_handover_estimate` | -2.1 (-3.6 to -0.4) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_no_time` | sampled | 2 | `lawnmower_sweep3_estimate` | -54.7 (-57.9 to -51.4) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_no_time` | sampled | 2 | `rule_sweep6_handover_estimate` | -0.4 (-1.8 to +1.2) | 99.6 | 0.996 / 0.988 | 99.9 | no |
| `sweep6_est_no_time` | sampled | 3 | `lawnmower_sweep3_estimate` | -56.4 (-59.3 to -53.3) | 99.9 | 0.998 / 0.990 | 100.0 | yes |
| `sweep6_est_no_time` | sampled | 3 | `rule_sweep6_handover_estimate` | -2.1 (-3.5 to -0.6) | 99.9 | 0.998 / 0.990 | 100.0 | yes |
| `sweep6_est_one` | deterministic | 1 | `lawnmower_sweep3_estimate` | -56.1 (-59.3 to -52.8) | 99.3 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_one` | deterministic | 1 | `rule_sweep6_handover_estimate` | -1.8 (-3.5 to +0.1) | 99.3 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_one` | deterministic | 2 | `lawnmower_sweep3_estimate` | -58.8 (-62.0 to -55.6) | 99.5 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_one` | deterministic | 2 | `rule_sweep6_handover_estimate` | -4.5 (-6.0 to -2.7) | 99.5 | 0.995 / 0.987 | 99.8 | no |
| `sweep6_est_one` | deterministic | 3 | `lawnmower_sweep3_estimate` | -51.8 (-55.2 to -48.4) | 99.2 | 0.994 / 0.986 | 99.7 | no |
| `sweep6_est_one` | deterministic | 3 | `rule_sweep6_handover_estimate` | +2.5 (+0.7 to +4.5) | 99.2 | 0.994 / 0.986 | 99.7 | no |
| `sweep6_est_one` | sampled | 1 | `lawnmower_sweep3_estimate` | -55.3 (-58.5 to -52.0) | 99.6 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est_one` | sampled | 1 | `rule_sweep6_handover_estimate` | -1.0 (-2.5 to +0.7) | 99.6 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est_one` | sampled | 2 | `lawnmower_sweep3_estimate` | -58.2 (-61.3 to -55.0) | 99.6 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est_one` | sampled | 2 | `rule_sweep6_handover_estimate` | -3.9 (-5.6 to -2.1) | 99.6 | 0.997 / 0.989 | 99.9 | no |
| `sweep6_est_one` | sampled | 3 | `lawnmower_sweep3_estimate` | -52.7 (-55.9 to -49.3) | 99.6 | 0.996 / 0.989 | 99.9 | no |
| `sweep6_est_one` | sampled | 3 | `rule_sweep6_handover_estimate` | +1.6 (+0.1 to +3.3) | 99.6 | 0.996 / 0.989 | 99.9 | no |

### Paired differences in AUV steps, each side the mean over its seeds

| Survey | Against | All episodes | 1 plume | 2 plumes | 3 plumes |
|---|---|---|---|---|---|
| `sweep6_est`, sampled | `rule_sweep6_handover_estimate`, scripted | -3.3 (-4.7 to -1.9) | -4.6 (-6.5 to -2.5) | -2.4 (-4.7 to +0.0) | -0.9 (-5.6 to +3.6) |
| `sweep6_est`, sampled | `rule_sweep6_scripted_gas_estimate`, scripted | -0.3 (-1.6 to +1.1) | -0.4 (-2.3 to +1.8) | -0.7 (-2.8 to +1.5) | +1.4 (-2.8 to +5.5) |
| `sweep6_est`, sampled | `lawnmower_sweep3_estimate`, scripted | -57.6 (-60.6 to -54.6) | -77.1 (-81.2 to -73.2) | -43.3 (-47.4 to -39.4) | -20.7 (-27.0 to -14.5) |
| `sweep6_est`, sampled | `sweep6_est_control`, sampled | -19.3 (-20.8 to -17.8) | -17.1 (-19.3 to -14.8) | -21.6 (-23.9 to -19.2) | -20.9 (-25.6 to -16.6) |
| `sweep6_est`, deterministic | `sweep6_est_control`, deterministic | -38.8 (-43.2 to -34.4) | -39.0 (-45.6 to -33.0) | -40.4 (-47.2 to -33.8) | -31.9 (-43.2 to -21.9) |
| `sweep6_est`, sampled | `sweep6_est_map`, sampled | -16.8 (-18.4 to -15.0) | -16.6 (-19.2 to -13.9) | -16.0 (-18.6 to -13.4) | -20.5 (-25.7 to -15.1) |
| `sweep6_est`, sampled | `sweep6_est_no_time`, sampled | -1.8 (-2.4 to -1.2) | -1.9 (-2.6 to -1.2) | -2.0 (-2.9 to -1.0) | -0.7 (-3.3 to +1.7) |
| `sweep6_est`, sampled | `sweep6_est_one`, sampled | -2.2 (-3.0 to -1.5) | -2.3 (-3.2 to -1.5) | -2.1 (-3.3 to -0.9) | -2.7 (-5.2 to -0.2) |
| `sweep6_est`, deterministic | `sweep6_est`, sampled | -0.2 (-0.9 to +0.5) | -0.1 (-0.9 to +0.8) | -0.1 (-1.3 to +1.1) | -1.1 (-3.3 to +1.0) |
| `sweep6_est_control`, deterministic | `sweep6_est_control`, sampled | +19.3 (+15.1 to +23.5) | +21.8 (+16.1 to +28.0) | +18.6 (+12.3 to +25.4) | +9.9 (-0.1 to +20.8) |

### Choices of the meta agent

| Survey | Actions | Search | Plume | Border | Search, early third | Search, middle third | Search, late third | Switches per 100 decisions |
|---|---|---|---|---|---|---|---|---|
| `lawnmower_sweep2_visit` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `lawnmower_sweep3_estimate` | scripted | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0 |
| `rule_sweep4_handover_visit` | scripted | 0.70 | 0.11 | 0.19 | 0.65 | 0.64 | 0.82 | 15.5 |
| `rule_sweep6_handover_estimate` | scripted | 0.68 | 0.09 | 0.23 | 0.61 | 0.60 | 0.82 | 14.9 |
| `rule_sweep6_scripted_gas_estimate` | scripted | 0.68 | 0.10 | 0.22 | 0.61 | 0.61 | 0.83 | 10.2 |
| `sweep4_visit` | deterministic | 0.71 | 0.11 | 0.19 | 0.65 | 0.65 | 0.82 | 6.9 ± 0.1 |
| `sweep4_visit` | sampled | 0.72 | 0.10 | 0.18 | 0.67 | 0.67 | 0.84 | 8.6 ± 0.3 |
| `sweep6_est` | deterministic | 0.69 | 0.09 | 0.22 | 0.63 | 0.62 | 0.82 | 8.3 ± 0.2 |
| `sweep6_est` | sampled | 0.69 | 0.09 | 0.22 | 0.63 | 0.63 | 0.82 | 9.4 ± 0.3 |
| `sweep6_est_control` | deterministic | 0.59 | 0.08 | 0.34 | 0.58 | 0.47 | 0.71 | 9.4 ± 0.5 |
| `sweep6_est_control` | sampled | 0.66 | 0.13 | 0.21 | 0.60 | 0.56 | 0.81 | 22.7 ± 0.2 |
| `sweep6_est_map` | deterministic | 0.73 | 0.23 | 0.04 | 0.92 | 0.75 | 0.52 | 6.2 ± 0.5 |
| `sweep6_est_map` | sampled | 0.72 | 0.21 | 0.07 | 0.92 | 0.71 | 0.53 | 11.3 ± 1.7 |
| `sweep6_est_no_time` | deterministic | 0.68 | 0.09 | 0.22 | 0.62 | 0.62 | 0.81 | 8.4 ± 0.3 |
| `sweep6_est_no_time` | sampled | 0.69 | 0.09 | 0.23 | 0.62 | 0.62 | 0.82 | 9.6 ± 0.2 |
| `sweep6_est_one` | deterministic | 0.68 | 0.08 | 0.24 | 0.63 | 0.62 | 0.81 | 8.3 ± 0.4 |
| `sweep6_est_one` | sampled | 0.68 | 0.08 | 0.24 | 0.62 | 0.62 | 0.81 | 9.7 ± 0.7 |
