# spacekit

![GitHub license](https://img.shields.io/github/license/spacetelescope/spacekit?color=black)
[![CodeFactor](https://www.codefactor.io/repository/github/spacetelescope/spacekit/badge)](https://www.codefactor.io/repository/github/spacetelescope/spacekit)
![Build Status](https://github.com/spacetelescope/spacekit/actions/workflows/ci.yml/badge.svg?branch=main)
[![Powered by STScI Badge](https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat)](http://www.stsci.edu)
[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14941394.svg)](https://doi.org/10.5281/zenodo.14941394)

Astronomical Data Science and Machine Learning Toolkit


![ML Dashboard](./previews/neural-network-graph.png)

## Setup

**Install with pip**

```bash
# install extra deps for all non-pipeline tools (analysis, training, data viz)
$ pip install spacekit[x]

# for bare-minimum dependencies (STScI/SDP pipeline operations):
$ pip install spacekit
```

**Install from source**

```bash
$ git clone https://github.com/spacetelescope/spacekit
$ cd spacekit
$ pip install -e .[x]
```

*Testing*

See `tox.ini` for a list of test suite markers.

```bash
# run all tests
$ pytest

# specify the `env` option to limit tests to a specific 'skope'
# env options: "svm", "hstcal", "jwstcal"
$ pytest --env svm -m svm
$ pytest --env hstcal -m cal
$ pytest --env jwstcal -m jwst
```


### Pre-Trained Neural Nets

**JWST Calibration Pipeline Resource Prediction (JWST)**

[JWST CAL Docs](https://spacekit.readthedocs.io/en/latest/skopes/jwst/cal.html)

* Inference ``spacekit.skopes.jwst.cal.predict``

*From the command line:*

```bash
$ python -m spacekit.skopes.jwst.cal.predict /path/to/inputs

# optionally specify a Program ID
$ python -m spacekit.skopes.jwst.cal.predict /path/to/inputs --pid 1076
```

*From python:*

```python
> from spacekit.skopes.jwst.cal.predict import JwstCalPredict
> input_path = "/path/to/level1/exposures"
# optionally specify a Program ID `pid` (default is None)
> jcal = JwstCalPredict(input_path, pid=1076)
> jcal.run_inference()
# estimations for L3 product memory footprints (GB) are stored in a dict under the `predictions` attribute. Ground truth values (latest actual footprints recorded) are shown as inline comments.
> jcal.predictions
{
    'jw01076-o101-t1_nircam_clear-f212n': {'gbSize': 10.02}, # actual: 10.553384 
    'jw01076-o101-t1_nircam_clear-f210m': {'gbSize': 8.72},  # actual: 11.196752
    'jw01076-o101-t1_nircam_clear-f356w': {'gbSize': 7.38}, # actual: 6.905737
}
# NOTE: the target number "t1" is not intended to match actual target IDs used by the pipeline.
```


**Single Visit Mosaic Alignment (HST)**

[SVM Docs](https://spacekit.readthedocs.io/en/latest/skopes/hst/svm.html)

* Preprocessing: ``spacekit.skopes.hst.svm.prep``
* Predict Image Alignments: ``spacekit.skopes.hst.svm.predict``
* Train Ensemble Classifier: ``spacekit.skopes.hst.svm.train``
* Generate synthetic misalignments†: ``spacekit.skopes.hst.svm.corrupt``
        
*† requires Drizzlepac*

**HST Calibration Pipeline Resource Prediction (HST)**

[HST CAL Docs](https://spacekit.readthedocs.io/en/latest/skopes/hst/cal.html)

* Training ``spacekit.skopes.hst.cal.train``
* Inference ``spacekit.skopes.hst.cal.predict``


**Exoplanet Detection with time-series photometry (K2, TESS)**

[K2 Docs](https://spacekit.readthedocs.io/en/latest/skopes/kepler/light-curves.html)

* ``spacekit.skopes.kepler.light_curves``


### Customizable Model Building Classes

Build, train and experiment with multiple model iterations using the ``builder.architect.Builder`` classes

Example: Build and train an MLP and 3D CNN ensemble network

- continuous/encoded data for the multi-layer perceptron
- 3 RGB image "frames" per image input for the CNN
- Stack mixed inputs and use the outputs of MLP and CNN as inputs for the final ensemble model

```python
ens = BuilderEnsemble(XTR, YTR, XTS, YTS, name="svm_ensemble")
ens.build()
ens.batch_fit()

# Save Training Metrics
outputs = f"data/{date_timestamp}"
com = ComputeBinary(builder=ens, res_path=f"{outputs}/results/test")
com.calculate_results()
```
# Load and plot metrics to evaluate and compare model performance

Analyze and compare results across iterations from metrics saved using ``analyze.compute.Computer`` class objects. Almost all plots are made using plotly and are dynamic/interactive.

```python
# Load data and metrics
from spacekit.analyzer.scan import MegaScanner
res = MegaScanner(perimeter="data/2022-*-*-*")
res._scan_results()
```

![ROC](./previews/roc-auc.png)

![Eval](./previews/model-performance.png)


### Preprocessing and Analysis Tools for Space Telescope Instrument Data

![box](./previews/eda-box-plots.png)

```python
from spacekit.analyzer.explore import HstCalPlots
res.load_dataframe()
hst = HstCalPlots(res.df, group="instr")
hst.scatter
```

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
        └── trained_networks
    └── dashboard
        └── cal
        └── svm
    └── datasets
        └── _base.py
        └── beam.py
        └── meta.py
    └── extractor
        └── load.py
        └── radio.py
        └── scrape.py
    └── generator
        └── augment.py
        └── draw.py
    └── logger
        └── log.py
    └── preprocessor
        └── encode.py
        └── ingest.py
        └── prep.py
        └── scrub.py
        └── transform.py
    └── skopes
        └── hst
            └── cal
                └── config.py
                └── predict.py
                └── train.py
                └── validate.py
            └── svm
                └── corrupt.py
                └── predict.py
                └── prep.py
                └── train.py
        └── jwst
            └── cal
                └── config.py
                └── predict.py
                └── train.py
        └── kepler
            └── light_curves.py
        
└── pyproject.toml
└── setup.cfg
└── tox.ini
└── tests
└── docker
└── docs
└── scripts
└── LICENSE
└── README.md
└── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
└── MANIFEST.in
└── bandit.yml
└── readthedocs.yaml
└── conftest.py
└── CHANGES.rst
```


```bash
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   

```
