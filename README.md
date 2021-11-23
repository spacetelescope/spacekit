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

### Classify new data using pre-trained model (from the command line):

1. Preprocess data (scrape from regression test json and fits files)

```bash
$ python -m spacekit.extractor.frame_data path/to/svmdata -o=./newdata.csv
```

2. Preprocess images (generate png preview files)

```bash
python -m spacekit.extractor.draw_mosaics path/to/svmdata -o=./img -d=./mydata.csv
```

3. Generate predictions

```bash
$ python -m spacekit.skopes.hst.mosaic.svm_predict ./mydata.csv ./img -m=./models/ensembleSVM -o=./results
```

### Build, train, evaluate new classifier from labeled data

Run steps 1 and 2 above, then:

```bash
$ python -m spacekit.skopes.hst.mosaic.svm_train ./mydata.csv ./img
```
