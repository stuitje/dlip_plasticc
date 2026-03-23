import avocado
import avocado.settings
import numpy as np

avocado.settings.load_settings("/home6/s4339150/Courses/avocado/avocado_settings.json")

# Load training set
dataset = avocado.load("plasticc_train")

# Subsample 5k objects
small = avocado.Dataset("plasticc_small", dataset.objects[:5000])

# Featurize directly — no augmentation
featurizer = avocado.plasticc.PlasticcFeaturizer()
small.extract_raw_features(featurizer)

# Save (avocado saves to features_directory from settings)
small.write_raw_features()

print("Done.")
print("Raw features shape:", small.raw_features.shape)
print("Classes:\n", small.metadata["target"].value_counts())