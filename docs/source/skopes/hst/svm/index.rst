.. _svm

***********************
spacekit.skopes.hst.svm
***********************

single visit mosaic alignment modeling

.. currentmodule:: spacekit.skopes.hst.svm

How-To
-------

* Generate Predictions (unlabeled data)
* Train and Evaluate Model (labeled data)
* Generate Synthetic Corruption Images (artificial corruption data)

.. toctree::
   :maxdepth: 1

   prep <prep.rst>
   predict <predict.rst>
   train <train.rst>
   corrupt <corrupt.rst>

Background Summary
------------------

Observations taken within a **single visit** are intended to represent a multi-wavelength view of objects in a given field-of-view. Most of the time, such observations are taken using pairs of guide stars to provide a very stable field-of-view throughout the entire visit resulting in images which **overlap almost perfectly**. Unfortunately, due to increasing limitations of the aging telescope systems, this is not always possible. The result is an increasing number of visits where observations:

   * **drift** and/or **roll** during the course of the visit
   * re-acquire at slightly **different pointings** from one orbit to the next 
   
Therefore, data across each visit cannot automatically be assumed to align. Single-visit mosaic (SVM) processing attempts to:

   * correct these relative alignment errors
   * align the data to an absolute astrometric frame 
   
In order to ensure that data across all filters used can be drizzled onto the same pixel grid.


Model Architecture
------------------

The SVM Classifier is a neural network trained on both "good" and "bad" (or valid and compromised) final drizzle products (images), along with statistical information produced during SVM processing quality analysis tests. The model's objective is to identify SVM total detection images that may have experienced an alignment error.

**Ensemble Classifier**

* Multi-Layer Perceptron (MLP)
* 3D Image CNN


Dataset
-------

- `SVM Regression test results`
- `SVM total detection images`


--- 

Setup
-----

**Install with pip**

.. code-block:: bash

    $ pip install spacekit


.. code-block:: bash

    $ git clone https://github.com/spacetelescope/spacekit
    $ cd spacekit
    $ pip install -e .


Run
---

**Example: HST Single Visit Mosaic Alignment Classification**

Classify new data using pre-trained model:


1. Preprocess data (scrape from regression test json and fits files, scrub/preprocess dataframe, generate png images for ML)

**from the command line**

.. code-block:: bash

    $ python -m spacekit.skopes.hst.svm.prep path/to/svmdata -f=svm_data.csv


**from python**

.. code-block:: python

    from spacekit.skopes.hst.svm.prep import run_preprocessing
    input_path = "/path/to/svm/datasets"
    run_preprocessing(input_path)
    
    # This is equivalent to using the default kwargs:
    run_preprocessing(input_path, h5=None, fname="svm_data", output_path=None, json_pattern="*_total*_svm_*.json", crpt=0)


Outputs:
* svm_data.csv
* raw_svm_data.csv
* svm_data.h5
* img/

2. Generate predictions

**from the command line**

.. code-block:: bash

    $ python -m spacekit.skopes.hst.svm.predict svm_data.csv img


**from python**

.. code-block:: python

    from spacekit.skopes.hst.svm.predict import predict_alignment
    data_file = "svm_data.csv"
    img_path = "img"
    predict_alignment(data_file, img_path)
    
    # This is equivalent to using the default kwargs:
    predict_alignment(data_file, img_path, model_path=None, output_path=None, size=None)


Outputs:
* predictions/clf_report.txt
* predictions/compromised.txt
* predictions/predictions.csv

----

Build, train, evaluate new classifier from labeled data
-------------------------------------------------------

Run step 1 (prep) above, then:

**from the command line**

.. code-block:: bash

    # Note: there are several option flags you can also include in this command
    $ python -m spacekit.skopes.hst.svm.train svm_data.csv img


**from Python**

.. code-block:: python

    # import spacekit training submodule
    from spacekit.skopes.hst.svm.train import run_training
    
    training_data = "svm_data.csv" # preprocessed dataframe (see step 1 above)
    img_path = "img" # preprocessed PNG image files (see step 1 above)
    
    run_training(training_data, img_path)
    
    # This is the same as using the default kwargs
    com, val = run_training(training_data, img_path, synth_data=None, norm=0, model_name=None, params=None, output_path=None)
    
    # Optional: view plots
    com.draw_plots()
    val.draw_plots()
