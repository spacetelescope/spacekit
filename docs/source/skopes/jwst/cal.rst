*****************************************************************
spacekit - JWST calibration pipeline resource prediction modeling
*****************************************************************

Inference
---------

Generate estimated memory footprints on unlabeled data.

.. toctree::
   :maxdepth: 2

   predict <cal/predict.rst>


Setup
-----

**Install with pip**

.. code-block:: bash

    $ pip install spacekit


.. code-block:: bash

    $ git clone https://github.com/spacetelescope/spacekit
    $ cd spacekit
    $ pip install -e .


Run Inference
-------------

***from the command line***

.. code-block:: bash

    $ python -m spacekit.skopes.jwst.cal.predict /path/to/inputs


***from python***

.. code-block:: python

    from spacekit.skopes.jwst.cal.predict import JwstCalPredict
    input_path = "/path/to/level1/exposures"
    jcal = JwstCalPredict(input_path)
    jcal.run_inference()
    jcal.predictions
    {
        'jw01076-o101-t1_nircam_clear-f212n': {'gbSize': 10.02},
        'jw01076-o101-t1_nircam_clear-f210m': {'gbSize': 8.72},
        'jw01076-o101-t1_nircam_clear-f356w': {'gbSize': 7.38},
    }

Outputs: dictionary of level 3 products and estimated memory footprint (GB)
