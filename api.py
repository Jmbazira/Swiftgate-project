import os
import uuid
import numpy as np
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# existing embedding function 
from embedding_face import get_face_embedding_safe

app = FastAPI(title="SwiftGate Hybrid Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 Research-Backed Threshold 
THRESHOLD = 0.65

UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def cosine_similarity(vec1, vec2):
    """Calculates the similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.get("/")
def health_check():
    return {"status": "SwiftGate Biometric Engine Online", "threshold": THRESHOLD}

@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(...), # Expected: Passport Scan
    image2: UploadFile = File(...), # Expected: Live Capture
    model_name: str = Form("Facenet") # Captures the dropdown value 
):
    path1 = os.path.join(UPLOAD_DIR, f"pass_{uuid.uuid4().hex}.jpg")
    path2 = os.path.join(UPLOAD_DIR, f"live_{uuid.uuid4().hex}.jpg")
    
    try:
        with open(path1, "wb") as buffer:
            shutil.copyfileobj(image1.file, buffer)
        with open(path2, "wb") as buffer:
            shutil.copyfileobj(image2.file, buffer)

        # 1. Generate Embeddings using the model name
        emb1 = get_face_embedding_safe(path1, model_name=model_name)
        emb2 = get_face_embedding_safe(path2, model_name=model_name)

        # 2. Check: was face found 
        if emb1 is None or emb2 is None:
            failed_source = "Passport" if emb1 is None else "Live Camera"
            raise HTTPException(
                status_code=400, 
                detail=f"Biometric Capture Failed: No face detected in {failed_source}."
            )

        # 3. Calculate Similarity
        similarity = float(cosine_similarity(emb1, emb2))
        verified = similarity >= THRESHOLD

        # 4. Return response to interface 
        return JSONResponse({
            "verified": verified,
            "similarity": similarity,
            "threshold": THRESHOLD,
            "model_used": model_name,
            "status": "MATCH_SUCCESS" if verified else "MATCH_DENIED"
        })

    except Exception as e:
        print(f"Operational Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

    finally:
        for path in [path1, path2]:
            if path and os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)