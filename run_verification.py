import numpy as np
import pandas as pd
from data_loader1 import load_utkface, assign_age_group
from pairs_generator4 import create_verification_pairs
from face_verification import verify_faces, cosine_similarity
from embedding_face import get_face_embedding_safe
from evaluation import evaluate_results

# 1. LOAD DATA
df = load_utkface("data/UTKFace")
df["age_group"] = df["age"].apply(assign_age_group)

# 2. GENERATE PAIRS
# 200 same + 200 different = 400 total
pairs = create_verification_pairs(df, num_pairs=200) 
results = []

print(f"🚀 Processing {len(pairs)} pairs. Extracting embeddings...")

# 3. EXTRACTION LOOP
for pair in pairs:
    img1, img2, label = pair['img1'], pair['img2'], pair['true_label']
    
    emb1 = get_face_embedding_safe(img1)
    emb2 = get_face_embedding_safe(img2)
    
    if emb1 is not None and emb2 is not None:
        # We only need the similarity score for the sweep
        sim = cosine_similarity(emb1, emb2)
        
        results.append({
            "similarity": sim,
            "true_label": label
        })

# 4. THRESHOLD SWEEP (The requested format)
def run_threshold_analysis(data_results):
    print("\n=== THRESHOLD ANALYSIS ===")
    # Sweep from 0.10 to 0.70 to find your 70-75% sweet spot
    thresholds = np.arange(0.10, 0.75, 0.05)
    
    for t in thresholds:
        metrics = evaluate_results(data_results, threshold=t)
        
        # This matches your requested formatting
        print(
            f"Threshold: {t:.2f} | "
            f"Accuracy: {metrics['accuracy']:.3f} | "
            f"FAR: {metrics['FAR']:.3f} | "
            f"FRR: {metrics['FRR']:.3f}"
        )

# Execute the analysis
if results:
    run_threshold_analysis(results)
else:
    print("❌ Error: No embeddings were generated. Check your paths.")