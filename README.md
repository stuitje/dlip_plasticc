# Avocado Reproduction

Reproduction of [Boone (2019)](https://doi.org/10.3847/1538-3881/ab5182) `avocado` photometric transient classifier on the PLAsTiCC dataset. Completed as part of the course Deep Learning in Physics at RUG.

## Results

| Metric | Boone (2019) | This work |
|--------|-------------|-----------|
| Flat-weighted log-loss | 0.468 | 0.528 |
| Redshift-weighted log-loss | 0.500 | 0.558 |
| Kaggle metric | 0.649 | 0.713 |

### Repository structure
```
avocado/ and scripts/ — patched avocado source (compatibility fixes)
notebooks/            — paper figures notebook and executed version with results
figures/              — reproduced figures as PDF
avocado_settings.json — settings file (update paths for your system)
```
All other files are unchanged from the original Boone (2019) repository.

We ran the code with Python 3.8.6 on the [Hábrók HPC cluster](https://wiki.hpc.rug.nl/habrok/introduction/start) using SLURM.

## About avocado

`avocado` is a general photometric classification code that is designed to
produce classifications of arbitrary astronomical transients and variable
objects. This code is designed from the ground up to address the problem of
biased spectroscopic samples. It does so by generating many lightcurves from
each object in the original spectroscopic sample at a variety of redshifts and
with many different observing conditions. The "augmented" samples of
lightcurves that are generated are much more representative of the full
datasets than the original spectroscopic samples.

The original codebase of `avocado` was developed for and won the [2018 Kaggle PLAsTiCC
challenge](https://kaggle.com/c/PLAsTiCC-2018). A paper describing the algorithms
implemented in this package can be found
[here](https://ui.adsabs.harvard.edu/abs/2019AJ....158..257B/abstract), and there is
also a discussion available on the [kaggle
forum](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033). The PLAsTiCC results
can be replicated using the latest version of `avocado` following the steps in the
documentation, and all of the code used to generate the figures in the paper can be
found in the `avocado_paper_figures.ipynb` notebook.

Carrick et al. 2021 (submitted to MNRAS) used `avocado` to study how to optimize
spectroscopic training samples for photometric classification of supernovae. An example
of the code used in that analysis can be found in the `spcc_augment_methods.ipynb`
notebook.

## Installation and Usage

Instructions on how to install and use `avocado` can be found on the [avocado
readthedocs page](https://avocado-classifier.readthedocs.io/en/latest/).

