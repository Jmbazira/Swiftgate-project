import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# =====================================================
# 1. PREPARE SCORES
# =====================================================
def extract_scores(results):
    """
    Extract labels and similarity scores from results.
    """
    y_true = np.array([r["true_label"] for r in results])
    y_scores = np.array([r["similarity"] for r in results])
    return y_true, y_scores


# =====================================================
# 2. ROC + AUC
# =====================================================
def plot_roc_curve(results):
    y_true, y_scores = extract_scores(results)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return fpr, tpr, thresholds, roc_auc


# =====================================================
# 3. EQUAL ERROR RATE (EER)
# =====================================================
def compute_eer(results):
    y_true, y_scores = extract_scores(results)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Find point where FAR ≈ FRR
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    print("\n=== EER ANALYSIS ===")
    print(f"EER: {eer:.4f}")
    print(f"EER Threshold: {eer_threshold:.4f}")

    return eer, eer_threshold


# =====================================================
# 4. BEST THRESHOLD (Youden's J)
# =====================================================
def find_best_threshold(results):
    y_true, y_scores = extract_scores(results)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print("\n=== BEST THRESHOLD (Youden's J) ===")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"TPR: {tpr[best_idx]:.4f}")
    print(f"FPR: {fpr[best_idx]:.4f}")

    return best_threshold


# =====================================================
# 5. AGE FAIRNESS ANALYSIS
# =====================================================
def age_group_fairness(results, threshold):
    """
    Requires each result to contain:
    - age_group1
    - age_group2

    If you only stored one age group earlier,
    adjust accordingly.
    """

    from collections import defaultdict

    group_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    for r in results:
        # 🔧 adjust if your structure differs
        group = r.get("age_group1", "unknown")

        pred = 1 if r["similarity"] >= threshold else 0
        true = r["true_label"]

        if pred == 1 and true == 1:
            group_stats[group]["TP"] += 1
        elif pred == 1 and true == 0:
            group_stats[group]["FP"] += 1
        elif pred == 0 and true == 0:
            group_stats[group]["TN"] += 1
        else:
            group_stats[group]["FN"] += 1

    print("\n=== AGE GROUP FAIRNESS ===")

    worst_frr = -1
    worst_group = None

    for group, s in group_stats.items():
        far = s["FP"] / max(1, (s["FP"] + s["TN"]))
        frr = s["FN"] / max(1, (s["FN"] + s["TP"]))

        print(f"\nGroup: {group}")
        print(f"  FAR: {far:.4f}")
        print(f"  FRR: {frr:.4f}")

        if frr > worst_frr:
            worst_frr = frr
            worst_group = group

    print("\n⚠️ Worst FRR group:", worst_group)
    print("Worst FRR value:", f"{worst_frr:.4f}")
