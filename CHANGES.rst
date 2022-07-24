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
