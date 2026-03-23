import avocado
import numpy as np

featurizer = avocado.plasticc.PlasticcFeaturizer()
num_chunks = 8

for chunk in range(5):
    print(f"\n--- Chunk {chunk}/5 ---")
    data = avocado.load("plasticc_train", chunk=chunk, num_chunks=num_chunks)
    data.extract_raw_features(featurizer)
    data.write_raw_features()
    print(f"Chunk {chunk} done. Shape: {data.raw_features.shape}")

print("\nAll chunks done.")