"""Configuration for JWST calibration reprocessing machine learning projects.
"""
GENKEYS = [
    "PROGRAM", # Program number
    "OBSERVTN", # Observation number
    "BKGDTARG", # Background target
    "VISITYPE", #  Visit type
    "TSOVISIT", # Time Series Observation visit indicator
    "TARGNAME", # Standard astronomical catalog name for target
    "TARG_RA", # Target RA at mid time of exposure
    "TARG_DEC", # Target Dec at mid time of exposure
    "INSTRUME", # Instrument used to acquire the data
    "DETECTOR", # Name of detector used to acquire the data
    "FILTER", # Name of the filter element used
    "PUPIL", # Name of the pupil element used  
    "EXP_TYPE", # Type of data in the exposure
    "CHANNEL", # Instrument channel 
    "SUBARRAY", # Subarray used
    "NUMDTHPT", # Total number of points in pattern
    "GS_RA",  #  guide star right ascension                     
    "GS_DEC", # guide star declination 
]
SCIKEYS = [
    "RA_REF",
    "DEC_REF",
    "CRVAL1",
    "CRVAL2",
]

COLUMN_ORDER = {
   "asn": [
      'nexposur',
      'bkgdtarg',
      'visitype',
      'tsovisit',
      'instrume',
      'detector',
      'filter',
      'pupil',
      'exp_type',
      'channel',
      'subarray',
      'numdthpt',
      'offset',
      'max_offset',
      'mean_offset',
      'sigma_offset',
      'err_offset',
      'sigma1_mean',
      'frac',
   ]
}


NORM_COLS = {"asn": [],}

RENAME_COLS = {"asn": [],}

X_NORM = {"asn": []}

