# Spatial ablation log

All model and seasonal-baseline metrics are calculated on the same test windows.

| Experiment | Spatial mode | Stations | K | Train/Val/Test | Best val MSE | Model MAE | Model RMSE | Baseline MAE | Baseline RMSE | Checkpoint |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| time_only_n20 | none | 20 | - | 65%/15%/20% | 0.028883 | 13.132179 | 41.152115 | 15.286173 | 32.923604 | models/ablation/time_only_n20.pth |
| gated_knn_n20 | gated_knn | 20 | 8 | 65%/15%/20% | 0.028838 | 13.175609 | 41.309192 | 15.286173 | 32.923604 | models/ablation/gated_knn_n20.pth |
| gat_lite_n20 | gat_lite | 20 | 8 | 65%/15%/20% | 0.028845 | 13.178141 | 41.299319 | 15.286173 | 32.923604 | models/ablation/gat_lite_n20.pth |
| gcn_lite_n20 | gcn_lite | 20 | 8 | 65%/15%/20% | 0.028858 | 13.174795 | 41.289475 | 15.286173 | 32.923604 | models/ablation/gcn_lite_n20.pth |
| a_gated_knn_full | gated_knn | 1913 | 8 | 65%/15%/20% | 0.225060 | 32.014950 | 71.247406 | 31.171948 | 68.426591 | models/ablation/a_gated_knn_full.pth |
