******************************************
spacekit - calcloud data pipeline modeling
******************************************

Calcloud Job Predict Data Dictionary
------------------------------------


.. toctree::
   :maxdepth: 2

   predict <cal/predict.rst>
   train <cal/train.rst>


DETECTOR
--------

**ACS**

* WFC (versus HRC or SBC) mosaic of two 2048 x 4096 pixels
* HRC and SBC are on 1024 x 1024

**WFC3**

* UVIS is a mosaic of two 4096 x 2051 pixels
* IR is 1024 x 1024
* could also use "NAXIS1" and "NAXIS2" which should reflect the size of the images.

- `detector = 1` : WFC, UVIS
- `detector = 0` : IR, HRC, SBC, et al 


DRIZCORR
--------

This will run the drizzling step which can take
a bit of time. Only applies to ACS and WFC3, so values for other instruments will show as NaN and be converted to zero (false).

- `drizcorr = 1` : PERFORM
- `drizcorr = 0` : OMIT, NaN

PCTECORR
--------

This keyword turns on the CTE (charge transfer efficiency) 
correction which is compute intensive processing. Only applies to ACS and WFC3.

- `pctecorr = 1` : PERFORM
- `pctecorr = 0` : OMIT, NaN


SUBARRAY
--------

The subarray readouts will be smaller than the full-frame
images and will process faster.

- `subarray = 1` : T
- `subarray = 0` : F

--- 

CRSPLIT
-------

2 (or at least a value greater than 1) 

This indicates multiple images to be used for cosmic
ray rejection, so multiple input images will be open
at the same time for processing.

- `crsplit = 0`
- `crsplit = 1`
- `crsplit = 2`


N_FILES (XFILES)
----------------

Total number of raw input files used in calibration. This feature is normalized and scaled into zero mean and unit variance values (`xfiles`) in order to stabilize variance and minimize skewness of the distribution.

TOTAL_MB (XSIZE)
----------------

Total size in megabytes of all raw files used in calibration.  This feature is normalized and scaled into zero mean and unit variance values (`xsize`) in order to stabilize variance and minimize skewness of the distribution.
