# Avocado

Photometric Classification of Astronomical Transients and Variables With Biased
Spectroscopic Samples

[![Documentation Status](https://readthedocs.org/projects/avocado-classifier/badge/?version=latest)](https://avocado-classifier.readthedocs.io/en/latest/?badge=latest)

## About

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

## Reproduce redshift-weighted classifier - no augmentation

I will assume you have downloaded the PLAsTiCC data and processed them into .h5 files. In the reproduction branch it is clear how to featurize the test set correctly to get the 10%. 

### Train (flat-weighted):
avocado_train_classifier plasticc_train flat_weight_noaug --class_weighting flat --object_weighting flat

### Train (redshift-weighted):
avocado_train_classifier plasticc_train redshift_weight_noaug --class_weighting flat --object_weighting redshift

### Prediction

Use following directly into terminal:
    python3 -c "
    import avocado
    import pandas as pd

    classifier = avocado.load_classifier('redshift_weight_noaug') 

    dataset = avocado.load('plasticc_test', metadata_only=True)

    features_path = avocado.settings['features_directory'] + '/features_v2_plasticc_test.h5'
    dataset.raw_features = pd.read_hdf(features_path, 'raw_features')

    predictions = dataset.predict(classifier)
    dataset.write_predictions()
    print('Done!')
    "

or flat_weight:
    python3 -c "
    import avocado
    import pandas as pd

    classifier = avocado.load_classifier('flat_weight_noaug')

    dataset = avocado.load('plasticc_test', metadata_only=True)

    features_path = avocado.settings['features_directory'] + '/features_v2_plasticc_test.h5'
    dataset.raw_features = pd.read_hdf(features_path, 'raw_features')

    predictions = dataset.predict(classifier)
    dataset.write_predictions()
    print('Done!')
    "

### Get results

After the prediction phase, you can just run avocado_paper_figures.ipynb fully, this will give you the necessary results.