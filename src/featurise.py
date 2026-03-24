import avocado
import numpy as np

featurizer = avocado.plasticc.PlasticcFeaturizer()

for chunk in range(8):  # all 8 chunks
    print(f"\n--- Chunk {chunk}/8 ---")
    data = avocado.load("plasticc_train", chunk=chunk, num_chunks=8)
    data.extract_raw_features(featurizer)
    data.write_raw_features()
    print(f"Chunk {chunk} done. Shape: {data.raw_features.shape}")

print("\nAll chunks done.")