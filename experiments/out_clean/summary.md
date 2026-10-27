# Experiment summary

Every number used in the paper, with the file in `experiments/out/` it came from.

## Step status

| step | status |
|---|---|
| dedup | ok (132s) |

## dedup

| key | value | source |
|---|---|---|
| images_total | 11346 | dedup_report.csv |
| images_train | 10053 | dedup_report.csv |
| images_val | 880 | dedup_report.csv |
| images_test | 413 | dedup_report.csv |
| leaked_val | 0 | dedup_report.csv |
| leaked_val_pct | 0.0 | dedup_report.csv |
| leaked_test | 0 | dedup_report.csv |
| leaked_test_pct | 0.0 | dedup_report.csv |
| pairs_near_phash_train-train | 1689 | dedup_summary.csv |
| pairs_near_phash_val-val | 20 | dedup_summary.csv |
| pairs_near_phash_test-test | 2 | dedup_summary.csv |
| pairs_near_phash_dihedral_val-val | 2 | dedup_summary.csv |
| source_groups_spanning_splits | 0 | dedup_report.csv |
| cross_crop_duplicate_pairs | 0 | dedup_report.csv |

## stats

| key | value | source |
|---|---|---|
| sugarcane_images | 3011 | class_totals.csv |
| sugarcane_boxes | 3529 | class_totals.csv |
| sugarcane_median_box_area_pct | 52.25 | class_totals.csv |
| wheat_images | 1774 | class_totals.csv |
| wheat_boxes | 2168 | class_totals.csv |
| wheat_median_box_area_pct | 27.44 | class_totals.csv |
| maize_images | 1200 | class_totals.csv |
| maize_boxes | 1200 | class_totals.csv |
| maize_median_box_area_pct | 98.29 | class_totals.csv |
| rice_images | 1440 | class_totals.csv |
| rice_boxes | 1667 | class_totals.csv |
| rice_median_box_area_pct | 48.44 | class_totals.csv |
| soybean_images | 3921 | class_totals.csv |
| soybean_boxes | 31997 | class_totals.csv |
| soybean_median_box_area_pct | 0.13 | class_totals.csv |
| train_images | 10053 | class_stats.csv |
| train_boxes | 35403 | class_stats.csv |
| train_background | 20 | class_stats.csv |
| val_images | 880 | class_stats.csv |
| val_boxes | 3456 | class_stats.csv |
| val_background | 2 | class_stats.csv |
| test_images | 413 | class_stats.csv |
| test_boxes | 1702 | class_stats.csv |
| test_background | 0 | class_stats.csv |
