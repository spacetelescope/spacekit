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
        └── scan.py
        └── track.py
    └── builder
        └── networks.py
    └── dashboard
        └── cal
            └── app.py
        └── svm
            └── app.py
    └── datasets
        └── hst_cal.py
        └── hst_svm.py
        └── k2_exo.py
    └── extractor
        └── load.py
        └── radio.py
        └── scrape.py
    └── generator
        └── augment.py
        └── draw.py
    └── preprocessor
        └── encode.py
        └── scrub.py
        └── transform.py
    └── skopes
        └── hst
            └── cal
                └── train.py
            └── svm
                    └── corrupt.py
                    └── predict.py
                    └── prep.py
                    └── train.py
        └── kepler
            └── light_curves.py
        └── trained_networks
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

1. Preprocess data (scrape from regression test json and fits files, scrub/preprocess dataframe, generate png images for ML)

***from the command line***

```bash
$ python -m spacekit.skopes.hst.svm.prep path/to/svmdata -f=svm_data.csv
```

***from python***

```python
from spacekit.skopes.hst.svm.prep import run_preprocessing
input_path = "/path/to/svm/datasets"
fname = run_preprocessing(input_path)
print(fname)
# svm_data.csv

# This is equivalent to using the default kwargs:
fname = run_preprocessing(input_path, h5=None, fname="svm_data", output_path=None, json_pattern="*_total*_svm_*.json", crpt=0, draw_images=1)
print(fname)
# default is "svm_data.csv"; customize filename and location using kwargs `fname` and `output_path`
```

Outputs:
* svm_data.csv
* raw_svm_data.csv
* svm_data.h5
* img/

2. Generate predictions

***from the command line***

```bash
$ python -m spacekit.skopes.hst.svm.predict svm_data.csv img
```

***from python***

```python
from spacekit.skopes.hst.svm.predict import predict_alignment
data_file = "svm_data.csv" # same as `fname` returned in `prep.py` above
img_path = "img" # default image foldername created above
predict_alignment(data_file, img_path)

# This is equivalent to using the default kwargs:
predict_alignment(data_file, img_path, model_path=None, output_path=None, size=None)
```

Outputs:
* predictions/
    * clf_report.txt
    * compromised.txt
    * predictions.csv

----

### Build, train, evaluate new classifier from labeled data

Run step 1 (prep) above, then:

***from the command line***

```bash
# Note: there are several option flags you can also include in this command
$ python -m spacekit.skopes.hst.svm.train svm_data.csv img
```

***from Python***

```python
# import spacekit training submodule
from spacekit.skopes.hst.svm.train import run_training

training_data = "svm_data.csv" # preprocessed dataframe (see step 1 above)
img_path = "img" # preprocessed PNG image files (see step 1 above)

run_training(training_data, img_path)

# This is the same as using the default kwargs
com, val = run_training(
    training_data, img_path, synth_data=None, norm=0, model_name=None, params=None, output_path=None
)

# Optional: view plots
com.draw_plots()
val.draw_plots()
```


```bash
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   

```