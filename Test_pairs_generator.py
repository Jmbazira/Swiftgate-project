from data_loader1 import load_utkface
from pairs_generator4 import create_verification_pairs

df = load_utkface("data/UTKFace")

pairs = create_verification_pairs(df, num_pairs=100)

print("Number of pairs:", len(pairs))
print("Sample pair:", pairs[0])
