import os
import numpy as np
from deepface import DeepFace

def get_face_embedding_safe(image_path: str, model_name: str = "Facenet") -> np.ndarray:
    """
    Extracts face features using the specified model architecture.
    Returns None if the file is missing, corrupt, or face detection fails.
    """
    
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        
        
        results = DeepFace.represent(
            img_path=image_path,
            model_name=model_name, 
            enforce_detection=False, 
            detector_backend='skip',  
            align=False
        )
        
        if results and len(results) > 0:
            
            return np.array(results[0]["embedding"])
        return None

    except Exception as e:
        
        print(f"Embedding Error with {model_name}: {str(e)}")
        return None