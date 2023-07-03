*****************************
spacekit.skopes.hst.cal.train
*****************************

.. currentmodule:: spacekit.skopes.hst.cal.train

.. automodule:: spacekit.skopes.hst.cal.train
   :members:


Spacekit HST "Calibration in the Cloud" (calcloud) Job Resource Allocation Model Training

This script imports and preprocesses job metadata for the Hubble Space Telescope data calibration pipeline,
which is then used as inputs to build, train and evaluate 3 neural networks for estimating AWS batch compute job resource requirements.

The networks include one multi-class classifier and two linear regression estimators. The classifier predicts which of 4 possible memory bin sizes (and therefore compute instance type) is most appropriate for reprocessing a given ipppssoot (i.e. "job"). The wallclock regressor estimates the maximum execution time ("wallclock" or "kill" time) in seconds needed to complete the job.

Memory Bin Classifier
---------------------
Allocating a job to a memory bin higher than required leads to unnecessary cost increases (because AWS billing tiers are directly related to compute instance types). However, if the estimated memory allocation is too low, the job will fail and have to be re-submitted at the next highest tier, also leading to increased costs. The majority of data being reprocessed for HST falls in the <2GB range, with only a handful in the highest tier (16-64GB).

Memory Regressor
----------------
Essentially identical to the classifier, the difference being that it returns a precise estimation value for memory in GB rather than a class. This is not needed for the pipeline (because it only needs to know which bin size) but we decided to keep the model for reference and analysis purposes.

Wallclock Regressor
-------------------
Estimates maximum execution or "kill" time in seconds it will take to complete the job. AWS Batch requires a minimum threshold of 60 seconds, with a large proportion of jobs completing below the one minute mark. Some jobs can take hours or even days to complete - if a job fails in memory after running for 24 hours, it would have to be re-submitted (huge cost). Likewise, if a job is allocated appropriate memory size but fails prematurely because it passes the maximum wallclock threshold, it would have to be resubmitted at a higher time allocation (huge cost). The relationship between memory needs and wallclock time is not linear, hence why there is a need for two separate models.

Ex:
python -m spacekit.skopes.hst.cal.train data/2021-11-04-1636048291

using this script produces a specific file output structure which can optionally be compressed and uploaded to s3.
The dataset used for training is updated with model predictions and can optionally be stored in a dynamodb table. 


.. code-block:: python
    
    """
    |- data/2021-11-04-1636048291
        |- data
            |- latest.csv
        |- models
            |- tx_data.json
            |- mem_clf
                |- {saved model binaries}
            |- mem_reg
                |- {saved model binaries}
            |- wall_reg
                |- {saved model binaries}
        |- results
            |- mem_bin {standard classifier filenames}
                |- acc_loss
                |- cmx
                |- cmx_norm
                |- fnfp
                |- history
                |- report
                |- roc_auc
                |- test_idx
                |- y_onehot
                |- y_pred
                |- y_scores
            |- memory {standard regressor filenames}
                |- history
                |- loss
                |- predictions
                |- residuals
                |- test_idx
            |- wallclock {standard regressor filenames}
                |- history
                |- loss
                |- predictions
                |- residuals
                |- test_idx
    """

examples:

download data from DynamoDB:

~/data/timestamp

upload to DDB: trained dataset with predictions CSV

~/data/timestamp/data/latest.csv

upload to s3: trained dataset with predictions CSV, results and models

~/data/timestamp/

Loading results from disk in a separate session
-----------------------------------------------

To load results from disk in a separate session (for plotting, analysis etc):

.. code-block:: python

    > bcom = ComputeMulti(res_path=f"{res_path}/mem_bin")
    > bin_out = bcom.upload()
    > bcom.load_results(bin_out)
    > test_idx = bin_out["test_idx"]
