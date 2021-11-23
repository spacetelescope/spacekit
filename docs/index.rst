**********************
spacekit documentation
**********************

This is the documentation for ``spacekit``,
Astronomical Data Science and Machine Learning Toolkit 

Reference/API
=============

.. automodapi:: spacekit

Contents:

.. toctree::
   :maxdepth: 2

   analyzer
   builder
   extractor
   preprocessor
   skopes


Skøpes: Machine Learning Applications
-------------------------------------

.. toctree::
   :maxdepth: 1

   HST <skopes/hst/svm-classifier.rst>
   K2/Kepler <skopes/kepler/planet-hunter.rst>

HST Data Processing Models
--------------------------

.. toctree::
   :maxdepth: 2

   Single Visit Mosaic Alignment Classifier <skopes/hst/svm-classifier.rst>

.. toctree::
   :maxdepth: 2

   svm_predict <skopes/hst/svm-predict.rst>
   svm_train <skopes/hst/svm-train.rst>
   svm_corrupt <skopes/hst/svm-corrupt.rst>


Analyzer
--------

.. toctree::
   :maxdepth: 1

   compute <analyzer/compute.rst>
   explore <analyzer/explore.rst>
   track <analyzer/track.rst>


Builder
-------

.. toctree::
   :maxdepth: 1

   cnn <builder/cnn.rst>


Extractor
---------

.. toctree::
   :maxdepth: 1

   draw_mosaics <extractor/draw_mosaics.rst>
   frame_data <extractor/frame_data.rst>
   load_images <extractor/load_images.rst>
   scrape_json <extractor/scrape_json.rst>


Preprocessor
------------

.. toctree::
   :maxdepth: 1

   augment <preprocessor/augment.rst>
   radio <preprocessor/radio.rst>
   transform <preprocessor/transform.rst>
