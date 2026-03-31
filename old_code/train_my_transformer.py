#!/usr/bin/env python3
import avocado
from avocado.classifier import TransformerClassifier
from avocado.sequence_featurizer import PlasticcSequenceFeaturizer

DATASET_NAME = "plasticc_augment"
CLASSIFIER_NAME = "my_transformer_seq"

featurizer = PlasticcSequenceFeaturizer(seq_len=350)

classifier = TransformerClassifier(
    name=CLASSIFIER_NAME,
    featurizer=featurizer,
    num_epochs=100,
    batch_size=64,
    lr=1e-4,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.2,
)

print(f"Loading dataset '{DATASET_NAME}'...")
dataset = avocado.load(DATASET_NAME)

print("Extracting sequence features...")
dataset.extract_raw_features(featurizer)

print(f"Training classifier '{CLASSIFIER_NAME}'...")
classifier.train(
    dataset,
    num_folds=5,
    random_state=42,
    val_fold=0,
    weight_decay=1e-2,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=2,
    min_lr=1e-6,
    early_stopping_patience=20,
    show_progress=True,
)

print(f"Best validation loss: {classifier.best_val_loss:.5f}")

if classifier.history is not None:
    print("\nTraining history (last 5 epochs):")
    print(classifier.history.tail())

print("Writing classifier...")
classifier.write(overwrite=True)

print("Done.")