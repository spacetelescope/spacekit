# spacekit

[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
![GitHub repo size](https://img.shields.io/github/repo-size/alphasentaurii/spacekit)
![GitHub license](https://img.shields.io/github/license/alphasentaurii/spacekit?color=black)


Astronomical Data Science and Machine Learning Toolkit


![ML Dashboard](./previews/neural-network-graph.png)

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

### Customizable Model Building Classes

```python
# Build and train stacked NN (MLP and 3D CNN)
ens = BuilderEnsemble(XTR, YTR, XTS, YTS, name="svm_ensemble")
ens.build()
ens.batch_fit()

# Save Training Metrics
com = ComputeBinary(builder=ens, res_path=f"{res_path}/test")
com.calculate_results()
```

### Pre-Trained Neural Nets

    * Calibration Data Pipeline (HST)

    * Single Visit Mosaic Alignment (HST)

    * Exoplanet Detection with time-series photometry (K2)


### ML Dashboard: Model Evaluation and Data Analysis

![ROC](./previews/roc-auc.png)

![Eval](./previews/model-performance.png)


### Preprocessing Tools for Space Telescope Data

![box](./previews/eda-box-plots.png)

![scatter](./previews/eda-scatterplots.png)


```python
spacekit
└── spacekit
    └── analyzer
        └── compute.py
        └── explore.py
        └── scan.py
        └── track.py
    └── builder
        └── architect.py
        └── blueprints.py
    └── dashboard
    └── datasets
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
            └── svm
                └── corrupt.py
                └── predict.py
                └── prep.py
                └── train.py
        └── kepler
        └── trained_networks
└── setup.py
└── tests
└── docker
└── LICENSE
└── README.md
```


```bash
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   

```