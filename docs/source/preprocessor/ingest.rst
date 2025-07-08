.. _ingest:

****************************
spacekit.preprocessor.ingest
****************************

.. currentmodule:: spacekit.preprocessor.ingest

Jwst Calibration Data Ingest
----------------------------

At STSCI, additional model training data is acquired daily from the telescope's calibration pipeline. Due to the nature of an automated 24-hour data collection cycle, some Level 3 products may still be processing at the time data is collected. This results in a given input file containing groups of L1 exposures with no matching L3 product. JwstCalIngest will run preprocessing on all L1 inputs and attempt to match them with an L3 product in the same file. 

- Any complete datasets (where a match is identified) are inserted into the "database", a file called `training.csv`. 
- Any remaining L1 exposures that did not find a match are stored into a separate "table" called `ingest.csv`. 

The next time this ingest process is run, the script will load both the new data as well as prior (unmatched) data. The assumption here is that the missing L3 product(s) (and sometimes even additional L1 exposures for this association) will eventually complete the pipeline and show up in subsequent files.

Additional output files are model-specific encoded subsets of `preprocessed` and `ingest`. Data is inserted into these in the same manner as appropriate. The actual files to be used for model training are named as `train-{modelname}.csv`, while `training.csv` contains all the original columns with unencoded values and is intended to be used primarily for data analysis and debugging purposes.

    - Database: {outpath}
    - Tables: {.csv files}
        - Accumulated data storing unencoded values
            - preprocessed:  complete L1-L3 groupings
            - ingest: unmatched L1 exposures
            - mosaics: c1XXX association candidate L3 products (currently not supported)
        - Encoded datasets finalized and ready for model training (input features + y-targets)
            - train-image: L3 image model
            - train-spec: L3 spectroscopy model
            - train-tac: L3 TSO/AMI/CORON model
        - Encoded input features of remaining L1 exposures (y-targets pending)
            - rem-image.csv
            - rem-spec.csv
            - rem-tac.csv


.. automodule:: spacekit.preprocessor.ingest
   :members:
