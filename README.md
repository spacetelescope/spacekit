# spacekit

[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
![GitHub repo size](https://img.shields.io/github/repo-size/alphasentaurii/spacekit)
![GitHub license](https://img.shields.io/github/license/alphasentaurii/spacekit?color=black)

Astronomical Data Science and Machine Learning Toolkit

```python
spacekit
└── spacekit
    └── analyzer
        └── compute.py
        └── explore.py
        └── track.py
    └── builder
        └── cnn.py
    └── extractor
        └── draw_mosaics.py
        └── frame_data.py
        └── load_images.py
        └── scrape_json.py
    └── preprocessor
        └── augment.py
        └── radio.py
        └── transform.py
    └── skopes
        └── hst
            └── mosaic
                    └── svm_corrupt.py
                    └── svm_predict.py
                    └── svm_train.py
                    └── models
└── setup.py
└── tests
└── LICENSE
└── README.md
```

## Setup

**Install with pip**

```bash
$ pip install spacekit
```

**Install from source**

```bash
$ git clone https://github.com/alphasentaurii/spacekit
$ cd spacekit
$ pip install -e .
```

## Run

**Example: HST Single Visit Mosaic Alignment Classification**

### Classify new data using pre-trained model:

1. Preprocess data (scrape from regression test json and fits files)

***from the command line***

```bash
$ python -m spacekit.extractor.frame_data path/to/svmdata -o=./newdata.csv
```


2. Preprocess images (generate png preview files)

```bash
python -m spacekit.extractor.draw_mosaics path/to/svmdata -o=./img
```

3. Generate predictions

```bash
$ python -m spacekit.skopes.hst.mosaic.svm_predict ./mydata.csv ./img -m=./models/ensembleSVM -o=./results
```

----

### Build, train, evaluate new classifier from labeled data

Run steps 1 and 2 above, then:

```bash
# Note: there are several option flags you can also include in this command
$ python -m spacekit.skopes.hst.mosaic.svm_train ./mydata.csv ./img
```

***Python***

```python
# import spacekit training submodule
from spacekit.skopes.hst.mosaic.svm_train import prep_ensemble_data, train_model, compute_results
# initialize some vars
training_data = "labeled_data.csv" # preprocessed dataframe (see step 1 above)
img_path = "./img" # preprocessed PNG image files (see step 2 above)
model_name = "my_new_model_name" # give the model a custom name (optional)
res_path = "./results" # where to save the model training results (optional)
# training parameters (optional - uses these defaults if none are explicitly set)
params=dict(
    batch_size=32,
    epochs=60, # set to 1 or 5 if just testing functionality...or 1000 if you have all day
    lr=1e-4,
    decay=[100000, 0.96],
    early_stopping=None,
    verbose=2,
    ensemble=True
    )
# create train test val splits, test-val index (for reviewing names of images model gets wrong)
tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = prep_ensemble_data(training_data, img_path)
# train the model
ens_model, ens_history = train_model(XTR, YTR, XTS, YTS, model_name)
# evaluate results (saved to local pickle files for later analysis)
compute_results(ens_model, ens_history, model_name, tv_idx, XTR, YTR, XTS, YTS, XVL, YVL)
```


```bash
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   

```