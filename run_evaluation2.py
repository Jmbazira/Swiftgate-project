from run_verification import results
from evaluation2 import (
    plot_roc_curve,
    compute_eer,
    find_best_threshold,
    age_group_fairness,
)

# 1. ROC
fpr, tpr, thresholds, auc_score = plot_roc_curve(results)

# 2. EER
eer, eer_threshold = compute_eer(results)

# 3. Best threshold
best_threshold = find_best_threshold(results)

# 4. Fairness at best threshold
age_group_fairness(results, best_threshold)
