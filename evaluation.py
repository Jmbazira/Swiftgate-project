def evaluate_results(results, threshold=0.5):
    TP = TN = FP = FN = 0

    for r in results:
        sim = r["similarity"]
        true = r["true_label"]

        #  recompute using new threshold
        pred = 1 if sim >= threshold else 0

        if pred == 1 and true == 1:
            TP += 1
        elif pred == 0 and true == 0:
            TN += 1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 1:
            FN += 1

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total else 0
    FAR = FP / (FP + TN) if (FP + TN) else 0
    FRR = FN / (FN + TP) if (FN + TP) else 0

    return {
        "accuracy": accuracy,
        "FAR": FAR,
        "FRR": FRR,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }
