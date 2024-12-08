1.1.2 (unreleased)
==================

docs
----
- Use furo as sphinx theme, improve page hierarchy and add custom domain [#104]
- downgrade numpy for compatibility with tensorflow in docs [#106]
- fix hyperlinks [#107]
- add custom landing page html template [#111]
- add overview page, dashboard images, and fix graphviz style for darkmode [#112]

preprocessor
------------
- explicitly pass `encoding=bytes` in transform.hypersonic_pliers for numpy 2 compatibility where this will no longer be the default for np.loadtxt [#92]

builder
-------
- Various minor fixes relating to CNN 2d model usage [#93]


1.1.1 (2024-07-11)
==================

installation
------------
- Set minimum python version to 3.10, include py312 in test suite, replace deprecated pkg_resources with importlib.metadata [#73]
- Set minimum tensorflow version to 2.16.1 [#73]
- Pre-trained neural networks updated for compatibility with Keras 3 [#73]
- Dockerfile images now use debian bookworm [#73]


1.1.0 (2024-04-09)
==================

new features
------------
- `builder.trained_networks` jwst_cal.zip includes updated (v2) `img3_reg` and new `spec3_reg` predictive models for image and spectroscopic data [#58]
- `preprocessor.ingest.JwstCalIngest` class and cmdline script for automated training data ingest [#57]
- `extractor.radio.JwstCalRadio` subclass for scraping datasets from MAST using ASN metadata [#51]
- `extractor.scrape.FitsScraper.scrape_dataframe` method added for scraping Fits data from dataframe [#52]

enhancements
------------

- `skopes.jwst.cal.predict` generates predictions for spectrosopic datasets in addition to image data. This update also allows further customization of user arguments: [#58]
    - `obs` to specify selection of a program ID + observation number
    - `input_path` accepts either a directory (default) or a filename. If filename, the script will try to find any input exposures that belong to the same program and observation number as that file.
    - `sfx` attribute is now customizable on instantiation of the class object (default is '_uncal.fits')
- `architect.builder.Builder.save_model` uses preferred keras archive format by default [#50]
- `preprocessor.transform.SkyTransformer` set offsets to 0 for gs/targ fiducial NaN values; custom filename for tx_file [#54]
- `preprocessor.prep.JwstCalPrep` updates in preparation for preprocessing spectroscopic data [#55]
    - revise spectroscopic data columns
    - save tx_file name with "-{expmode}" to differentiate between image and spec normalization params
    - rename target attributes: y_img_train, y_img_test to y_reg_train, y_reg_test
- `preprocessor.scrub.JwstCalScrubber` more sophisticated exposure grouping and L3 product naming [#56]

bug fixes
---------
- `preprocessor.encode.PairEncoder.handle_unknowns` create single new encoding value per unidentified variable [#53]



1.0.1 (2024-04-03)
==================

bugfixes
--------

- move HstSvmRadio import inside class method to avoid importing astroquery unnecessarily [#49]

- temporarily pin tf max version to 2.15 to ensure compatibility with models saved in 2.13 or older

- matplotlib style setting looks for "seaborn-v0_8-bright" if "seaborn-bright" unavailable, fallback uses default style


installation / automation
-------------------------

- GA workflow minor revision: pypi publish [#46]

- Replace flake8 with ruff, replace deprecated tf.keras.wrappers.scikit_learn with scikeras, add GA workflows [#45]

documentation
-------------

- Update readthedocs.yaml for compatibility with latest formatting requirements [#44]

- RTD: Install graphviz before building docs [#47]


1.0.0 (2023-08-10)
==================

- New feature: JWST Calibration Processing resource prediction model and skope (prediction script) added under the architecture name "jwst_cal"

- Pretrained neural network files and paths renamed: calmodels.zip is now hst_cal.zip,
ensemble.zip is now svm_align.zip.

- If keras models are saved using the older SavedModel format, you must pass `keras_archive=False` when loading a saved model. By default, new models will be saved using the newer keras archive format.

- Tests added for JWST; existing tests and metadata updated to reflect above changes

- Updated zenodo version ID for remote test data


0.4.1 (2023-07-03)
==================

- bugfix set dataframe columns with bracket instead of curly bracket (resolves pandas>1.4 incompatibility)

- remove pandas pinned version

- improved log handling with spacekit/logger module

- added predict script for hst cal skope

- updated docker dashboard templates

- enhancements for loading pretrained models

- pytest configuration updates and new tests added

- plugin for external test data

- updated repo url badges

- updated documentation


0.4.0 (2022-12-08)
==================

- bugfix scikit-learn replaces deprecated sklearn dependency

- temporarily pinned `pandas` dependency to 1.4.x and below due to column setting bug in v1.5

- bugfix keras `load_img` method imported from tf.keras.preprocessing.image instead of tf.keras.utils

- new feature skopes.hst.cal model training, inference, cross-validation scripts added

- new feature svm dashboard predict view

- svm ensemble model archive file `ensembleSVM.zip` renamed as `ensemble.zip`. This extracts to `models/ensemble/` with `tx_data.json` (transform data) and `ensembleSVM` (keras model binaries) inside of the `ensemble/` parent directory. Previously, the json file was inside ensembleSVM alongside the binaries.


0.3.2 (2022-07-24)
==================

- Docker image deployment bugfixes and cleaner organization

- Updated calcloud model results formatting to conform with spacekit compute module I/O

- Bugfix for dataset scrape/import


0.3.1 (2022-05-04)
==================

- Bug fix relating to the SVM predict.py Classification Report which mistakenly assumed all categorical types are represented in the data (not necessarily the case for prediction inputs). Fixing the encoder resolves the issue (see below)

- A custom encoder class `PairEncoder` was created, allowing a user to pass in explicit key-pair values (a dictionary) for categorical features and `SvmEncoder` was updated to use this for encoding “category" (scene/field), "detector" and "wcs".

- Additional tests added to test_encode.py for the above case

- Minor enhancements to SVM classification report for better readability.


0.3.0 (2022-02-16)
==================

- SVM module added to `skopes` for evaluating the alignment of HST Single Visit Mosaic images using an "ensembled" 4D image classifier and MLP model.
- CAL dashboard enhancements
- new feature SVM dashboard for model evaluation and data analysis
- enhancements to SVM prep, predict and training modules
- significant additions made to pytest test suite for primary svm-related modules
- minor bug fixes and enhancements
- ability to load/save image arrays as compressed numpy files (single .npz file instead of individual pngs).
- load dataset module added for calcloud dashboard
- Read the Docs documentation and API
