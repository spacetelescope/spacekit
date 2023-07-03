"""Configuration for HST calibration reprocessing machine learning projects.
ASN = standard pipeline ASN data
SVM = single visit mosaic data
"""

COLUMN_ORDER = {
    "asn": [
        "n_files",
        "total_mb",
        "drizcorr",
        "pctecorr",
        "crsplit",
        "subarray",
        "detector",
        "dtype",
        "instr",
    ],
    "svm": [],
}

NORM_COLS = {"asn": ["n_files", "total_mb"], "svm": []}

RENAME_COLS = {"asn": ["x_files", "x_size"], "svm": []}

X_NORM = {
    "asn": [
        "x_files",
        "x_size",
        "drizcorr",
        "pctecorr",
        "crsplit",
        "subarray",
        "detector",
        "dtype",
        "instr",
    ]
}
