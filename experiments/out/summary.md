# Experiment summary

Every number used in the paper, with the file in `experiments/out/` it came from.

## Step status

| step | status |
|---|---|
| dedup | ok (199s) |

## dedup

| key | value | source |
|---|---|---|
| images_total | 11346 | dedup_report.csv |
| images_train | 9697 | dedup_report.csv |
| images_val | 1098 | dedup_report.csv |
| images_test | 551 | dedup_report.csv |
| leaked_val | 223 | dedup_report.csv |
| leaked_val_pct | 20.31 | dedup_report.csv |
| leaked_test | 138 | dedup_report.csv |
| leaked_test_pct | 25.05 | dedup_report.csv |
| pairs_near_phash_train-train | 1345 | dedup_summary.csv |
| pairs_near_phash_train-val | 176 | dedup_summary.csv |
| pairs_near_phash_test-train | 106 | dedup_summary.csv |
| pairs_near_phash_val-val | 34 | dedup_summary.csv |
| pairs_near_phash_test-val | 39 | dedup_summary.csv |
| pairs_near_phash_test-test | 11 | dedup_summary.csv |
| pairs_near_phash_dihedral_train-val | 525 | dedup_summary.csv |
| pairs_near_phash_dihedral_test-val | 18 | dedup_summary.csv |
| pairs_near_phash_dihedral_val-val | 6 | dedup_summary.csv |
| pairs_near_phash_dihedral_test-train | 341 | dedup_summary.csv |
| pairs_near_phash_dihedral_test-test | 9 | dedup_summary.csv |
| pairs_same_source_stem_train-val | 37 | dedup_summary.csv |
| pairs_same_source_stem_test-train | 16 | dedup_summary.csv |
| pairs_same_source_stem_test-val | 2 | dedup_summary.csv |
| source_groups_spanning_splits | 40 | dedup_report.csv |
| cross_crop_duplicate_pairs | 0 | dedup_report.csv |

## stats

| key | value | source |
|---|---|---|
| sugarcane_images | 3011 | class_totals.csv |
| sugarcane_boxes | 3529 | class_totals.csv |
| sugarcane_median_box_area_pct | 52.69 | class_totals.csv |
| wheat_images | 1774 | class_totals.csv |
| wheat_boxes | 2168 | class_totals.csv |
| wheat_median_box_area_pct | 27.71 | class_totals.csv |
| maize_images | 1200 | class_totals.csv |
| maize_boxes | 1200 | class_totals.csv |
| maize_median_box_area_pct | 98.29 | class_totals.csv |
| rice_images | 1440 | class_totals.csv |
| rice_boxes | 1667 | class_totals.csv |
| rice_median_box_area_pct | 47.24 | class_totals.csv |
| soybean_images | 3921 | class_totals.csv |
| soybean_boxes | 31997 | class_totals.csv |
| soybean_median_box_area_pct | 0.13 | class_totals.csv |
| train_images | 9697 | class_stats.csv |
| train_boxes | 34810 | class_stats.csv |
| train_background | 19 | class_stats.csv |
| val_images | 1098 | class_stats.csv |
| val_boxes | 3813 | class_stats.csv |
| val_background | 3 | class_stats.csv |
| test_images | 551 | class_stats.csv |
| test_boxes | 1938 | class_stats.csv |
| test_background | 0 | class_stats.csv |
