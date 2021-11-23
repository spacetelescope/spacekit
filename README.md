# spacekit
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

**Example: HST Mosaic Alignment Classification**

### Classify new data using pre-trained model (from the command line):

1. Preprocess data (scrape from regression test json and fits files)

```bash
$ python -m spacekit.extractor.frame_data h5filename -d=path/to/svmdata -o=./newdata.csv
```

2. Preprocess images (generate png preview files)

```bash
python -m spacekit.extractor.draw_mosaics path/to/svmdata -o=./img -d=./mydata.csv
```

3. Generate predictions

```bash
$ python -m spacekit.skopes.hst.mosaic.svm_predict ./mydata.csv ./img -m=./models/ensembleSVM -o=./results
```
