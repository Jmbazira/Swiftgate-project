import numpy as np
from evaluation import evaluate_results

def threshold_sweep(results):
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    print("\n=== THRESHOLD ANALYSIS ===")
    for t in thresholds:
        metrics = evaluate_results(results, threshold=t)
        print(f"Threshold: {t:.2f} | Accuracy: {metrics['accuracy']:.3f} | FAR: {metrics['FAR']:.3f} | FRR: {metrics['FRR']:.3f}")