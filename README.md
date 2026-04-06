# DLiP_PLAsTiCC
Repository for final project of the course Deep Learning in Physics. It contains extensions for photometric classification of astronomical transients, built on top of the [avocado](https://github.com/kboone/avocado) framework for the PLAsTiCC dataset. 

This repository implements three classifiers as alternatives to or improvements over the gradient-boosted decision tree in Boone (2019):

- **Residual MLP** trained on avocado GP features
- **Transformer** trained directly on raw augmented light curves
- **Transformer** trained on GP-interpolated light curves

Other attempted pipelines and models are also present, but do not necessarily perform well, or run from end to end. 

## Project structure


## Installation

**Requirements: Python 3.10 or higher.**

### 1. Install avocado

> **Note:** It is advised to use a Conda or virtual environment to store all dependencies. For example:
> ```bash
> conda create -n plasticc python=3.10 -y
> ```

Avocado must be installed manually. To avoid any compatibility conflicts, please install this fork:
```bash
pip install git+https://github.com/stuitje/avocado.git
cd avocado
pip install -e .
```

### 2. Install this package
```bash
git clone https://github.com/stuitje/dlip_plasticc.git
cd dlip_plasticc
pip install -e .
```

## Usage

To use this code, the necessary data must first be downloaded, preprocessed, augmented and, depending on the user's choice of model, featurised, using Avocado's pipeline. 

Following the instructions on the [avocado documentation](https://avocado-classifier.readthedocs.io/en/latest/): 

### 1. Set up a working directory

Create and move to a new working directory for avocado. All of the datasets, classifiers and predictions will be stored in this directory, so consider avoiding your homefolder if your on a HPC cluster (use, for example, a `scratch/` directory). For example:

```bash
mkdir ~/plasticc
```

### 2. Update the avocado settings file 

Update `avocado_settings.json` with the appropriate paths to the directory (created above), where you store your data, predictions, classifiers and features.

### 3. Download the PLAsTiCC dataset

This downloads the PLAsTiCC dataset from [zenodo](https://zenodo.org/records/2539456), and preprocesses it automatically.

```bash
cd ~/plasticc # or any other folder name
avocado_download_plasticc
```

### 4. Augment the PLAsTiCC dataset

Here the data is augmented to create a number of redshift realisations of each object. Adjust the number of augmentations as wished. Our results are based on 15 augmentations; Avocado uses 100. This can take a couple of hours to run.

> **Note**: If you are on a HPC cluster, it is recommended to run this in e.g. a SLURM job.

```bash
avocado_augment plasticc_train plasticc_augment \
    --num_augments 15 \
    --num_chunks 100 \
```

### 5. Featurise the PLAsTiCC dataset [optional]

To use e.g. the MLP on the featurised dataset, featurisation is necessary. This will take a long time (full test set, sequentially, ~100 hours), especially if you used many augmentations in the step above. Consider parallelising, and submit it as a SLURM job when on a HPC cluster. Consider featurising only a part of the test set, at least in first instance. 

```
avocado_featurize plasticc_train # usually not necessary, as we use the augmented dataset
avocado_featurize plasticc_test
avocado_featurize plasticc_augment
```

To featurise only a part (e.g. 10%) of the test set:

```
for chunk in $(seq 0 49); do
    avocado_featurize plasticc_test \
        --num_chunks 500 \
        --chunk $chunk
done
```

### 6. GP-fit the training and test set [optional]

To use a transformer on GP-fitted light curves, the raw light curves must 
first be interpolated onto a uniform time grid using GP regression. This 
step is optional and only required if you want to train the GP transformer.

The GP fitting is parallelised across chunks using SLURM job arrays. 
Each chunk is processed independently and written to its own `.h5` file, 
which are then merged into a single file at the end.

**Training set:**
```bash
sbatch jobs/submit_chunks.sh
```

Once all jobs have completed, check for missing chunks and merge:
```bash
# check all chunks are present
ls /scratch/.../data/chunks/chunk_*.h5 | wc -l  # should print 100

# merge into a single file
scripts/merge_chunks
```

**Test set** (first 100 chunks = 20% of test set):
```bash
sbatch jobs/submit_chunks_test.sh

# merge when done
scripts/ merge_chunks  # update CHUNK_DIR and NUM_CHUNKS in the script first
```

> **Note:** GP fitting the full training set takes approximately 12 core 
> hours in total, but only ~5 minutes of wall time when parallelised across 
> 100 SLURM jobs. If you do not have access to a SLURM cluster, you can 
> run the chunks sequentially using `scripts/run_chunk` and 
> `scripts/run_chunk_test` directly.

### 7. Configure 

Default model settings are in `configs/default.toml`. An `avocado_settings.json` file must be present in the working directory when running notebooks or scripts. 

> **Note:** In most notebooks, the default configuration parameters are overridden, so updating this file is not needed. 

### 8. Run a notebook

Training and evaluation notebooks are in the root directory:

| Notebook | Description |
|---|---|
| `train_mlp.ipynb` | Train the residual MLP on avocado features |
| `train_transformer.ipynb` | Train the transformer on raw light curves |
| `train_transformer_gp.ipynb` | Train the transformer on GP-interpolated light curves |

## Repository

[https://github.com/stuitje/dlip_plasticc](https://github.com/stuitje/dlip_plasticc)

## Reference

Boone, K. (2019). Avocado: Photometric Classification of Astronomical 
Transients with Gaussian Process Augmentation. *The Astronomical Journal*, 
158(6), 257. https://doi.org/10.3847/1538-3881/ab5182
