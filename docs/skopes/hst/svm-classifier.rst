***************************************************
spacekit - single visit mosaic alignment classifier
***************************************************

How-To
-------

* Generate Predictions (unlabeled data)
* Train and Evaluate Model (labeled data)
* Generate Synthetic Corruption Images (artificial corruption data)

.. toctree::
   :maxdepth: 2

   svm_predict <svm-predict.rst>
   svm_train <svm-train.rst>
   svm_corrupt <svm-corrupt.rst>

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

