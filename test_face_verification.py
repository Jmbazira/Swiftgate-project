from embedding_face import get_face_embedding
from face_verification import verify_faces

img1 = "data/UTKFace/29_1_1_20170112235112993.jpg.chip.jpg"
img2 = "data/UTKFace/29_1_1_20170112235112993.jpg.chip.jpg"

emb1 = get_face_embedding(img1)
emb2 = get_face_embedding(img2)

label, score = verify_faces(emb1, emb2)

print("Prediction:", label)
print("Similarity:", score)
