*****************************************************************
spacekit - JWST calibration pipeline resource prediction modeling
*****************************************************************

How-To
-------

* Generate Predictions (unlabeled data)

.. toctree::
   :maxdepth: 2

   predict <cal/predict.rst>


## Setup

**Install with pip**

.. code-block:: bash

    $ pip install spacekit


.. code-block:: bash

    $ git clone https://github.com/spacetelescope/spacekit
    $ cd spacekit
    $ pip install -e .


## Run

1. Generate predictions

***from the command line***

.. code-block:: bash

    $ python -m spacekit.skopes.jwst.cal.predict /path/to/inputs


***from python***

.. code-block:: python

    from spacekit.skopes.jwst.cal.predict import JwstCalPredict
    input_path = "/path/to/level1/exposures"
    jcal = JwstCalPredict(input_path)
    jcal.run_inference()

Outputs:
* jcal.predictions : dictionary of level 3 products and estimated memory footprint (GB)
