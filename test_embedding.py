from data_loader1 import load_utkface
from embedding_face import get_face_embedding

df = load_utkface("data/UTKFace")

sample_image = df.iloc[0]["image_path"]

embedding = get_face_embedding(sample_image)

print("Embedding length:", len(embedding))
print("First 5 values:", embedding[:5])
