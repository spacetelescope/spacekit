**********************
spacekit documentation
**********************

This is the documentation for ``spacekit``,
the Astronomical Data Science and Machine Learning Toolkit 


Overview
========

Spacekit is a python library designed to do the heavy lifting of machine learning in astronomy-related applications.

The modules contained in this package can be used to assist and streamline each step of a typical data science project:

1. :doc:`Ingest/Extract  <extractor/index>` import large datasets from a variety of file formats .csv, .hdf5, .fits, .json, .png (.asdf coming soon)

2. :doc:`Scrub/Preprocess <preprocessor/index>` scrub and preprocess raw data to prepare it for use in a machine learning model

3. :doc:`Modeling <builder/index>` build, train and deploy custom machine learning models using classification, logistic regression estimation, computer vision and more

4. :doc:`Analysis <analyzer/index>` evaluate model performance and do exploratory data analysis (EDA) using interactive graphs and visualizations

5. :doc:`Visualize <dashboard/index>` deploy a web-based custom dashboard for your models and datasets via docker, a great way to summarize and share comparative model evaluations and data analysis visuals with others


Applications
------------

The :doc:`Skøpes <skopes/index>` module includes real-world machine learning applications used by the Hubble and James Webb Space Telescopes in data calibration pipelines. These mini-applications are an orchestration of functions and classes from other spacekit modules to run automated analysis, training, and inference in real-time on a local server or in the cloud (AWS).


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
