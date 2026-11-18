# Participant-level out-of-fold prediction summary

- Selected model: **svm_rbf**
- Evaluation: **leave-one-participant-out**; each listed participant was held out during prediction.
- Participants included: **13**
- Condition-summary rows: **26**
- Windows summarized: **1103**

These are experimental baseline/stress condition classifications, not clinical diagnoses. The metric below is pooled over out-of-fold windows; it is not external validation.

## Pooled out-of-fold metrics

- Accuracy: **0.534**
- Macro precision: **0.563**
- Macro recall: **0.567**
- Macro F1: **0.532**
- ROC-AUC: **0.542**
- Confusion matrix (actual rows / predicted columns; baseline, stress): `[[261, 124], [390, 328]]`

## Predictions by participant and recorded condition

`window_accuracy` is the fraction of windows in that recorded condition predicted as its corresponding configured label.

| Participant | Recorded condition | Windows | Correct | Accuracy | Predicted baseline | Predicted stress | Majority output | Mean stress probability |
|---|---|---:|---:|---:|---:|---:|---|---:|
| ananya_d | baseline | 22 | 22 | 1.000 | 22 | 0 | baseline | 0.396 |
| ananya_d | stress | 43 | 0 | 0.000 | 43 | 0 | baseline | 0.410 |
| Chaitanya | baseline | 29 | 9 | 0.310 | 9 | 20 | stress | 0.624 |
| Chaitanya | stress | 73 | 73 | 1.000 | 0 | 73 | stress | 0.811 |
| Divyansh | baseline | 17 | 17 | 1.000 | 17 | 0 | baseline | 0.383 |
| Divyansh | stress | 123 | 91 | 0.740 | 32 | 91 | stress | 0.645 |
| Hema2 | baseline | 58 | 50 | 0.862 | 50 | 8 | baseline | 0.510 |
| Hema2 | stress | 43 | 20 | 0.465 | 23 | 20 | baseline | 0.608 |
| KanavR | baseline | 29 | 5 | 0.172 | 5 | 24 | stress | 0.722 |
| KanavR | stress | 37 | 32 | 0.865 | 5 | 32 | stress | 0.808 |
| MadanR | baseline | 28 | 24 | 0.857 | 24 | 4 | baseline | 0.417 |
| MadanR | stress | 45 | 24 | 0.533 | 21 | 24 | stress | 0.539 |
| Madhusudhan | baseline | 14 | 12 | 0.857 | 12 | 2 | baseline | 0.390 |
| Madhusudhan | stress | 76 | 21 | 0.276 | 55 | 21 | baseline | 0.501 |
| ManojaS | baseline | 27 | 0 | 0.000 | 0 | 27 | stress | 0.938 |
| ManojaS | stress | 44 | 37 | 0.841 | 7 | 37 | stress | 0.801 |
| NithinG | baseline | 27 | 27 | 1.000 | 27 | 0 | baseline | 0.158 |
| NithinG | stress | 40 | 0 | 0.000 | 40 | 0 | baseline | 0.078 |
| PrathamS | baseline | 25 | 1 | 0.040 | 1 | 24 | stress | 0.873 |
| PrathamS | stress | 2 | 2 | 1.000 | 0 | 2 | stress | 0.763 |
| shreeshK | baseline | 31 | 31 | 1.000 | 31 | 0 | baseline | 0.189 |
| shreeshK | stress | 33 | 0 | 0.000 | 33 | 0 | baseline | 0.411 |
| SnehalShetty | baseline | 19 | 19 | 1.000 | 19 | 0 | baseline | 0.165 |
| SnehalShetty | stress | 93 | 0 | 0.000 | 93 | 0 | baseline | 0.163 |
| vasavi_naik | baseline | 59 | 44 | 0.746 | 44 | 15 | baseline | 0.539 |
| vasavi_naik | stress | 66 | 28 | 0.424 | 38 | 28 | baseline | 0.590 |

## Excluded from condition-labelled prediction summary

These recordings have missing or duplicate condition markers, so their windows cannot be assigned reliable baseline/stress ground truth.

- **aarya**: duplicate Baseline_start at source lines [2, 26064]; manual review required
- **Hema_sunkad**: missing Baseline_start; missing Baseline_end; missing stress_end; recording is not eligible in dataset manifest
- **Meghamam**: missing Baseline_start; missing stress_start; missing stress_end; recording is not eligible in dataset manifest
- **Nikita_halli**: missing Baseline_start; missing Baseline_end; missing stress_end; recording is not eligible in dataset manifest
- **PavitraH**: duplicate Baseline_start at source lines [2, 86916, 89812]; manual review required; duplicate Baseline_end at source lines [20960, 129616]; manual review required
- **Rajeshwari**: missing stress_start; recording is not eligible in dataset manifest
- **VAISHNAVI**: missing stress_start; missing stress_end; recording is not eligible in dataset manifest
- **YashS**: duplicate stress_start at source lines [60516, 138714]; manual review required; duplicate stress_end at source lines [101656, 139314]; manual review required
