import numpy as np

def cosine_similarity(emb1, emb2):
    """
    Calculates how similar two faces are. 
    1.0 = Perfect Match | 0.0 = Completely Different
    """
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    # Normalize vectors 
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    emb1 = emb1 / norm1
    emb2 = emb2 / norm2

    
    return float(np.dot(emb1, emb2))

def verify_faces(emb1, emb2, threshold=0.30):
    """
    Returns the binary prediction and the raw similarity score.
    """
    sim = cosine_similarity(emb1, emb2)
    
    
    prediction = 1 if sim >= threshold else 0
    
    return prediction, sim