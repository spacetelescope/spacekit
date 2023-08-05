"""Configuration for JWST calibration reprocessing machine learning projects.
"""
GENKEYS = [
    "PROGRAM",  # Program number
    "OBSERVTN",  # Observation number
    "BKGDTARG",  # Background target
    "IS_IMPRT"  # NIRSpec imprint exposure 
    "VISITYPE",  # Visit type
    "TSOVISIT",  # Time Series Observation visit indicator
    "TARGNAME",  # Standard astronomical catalog name for target
    "TARG_RA",  # Target RA at mid time of exposure
    "TARG_DEC",  # Target Dec at mid time of exposure
    "INSTRUME",  # Instrument used to acquire the data
    "DETECTOR",  # Name of detector used to acquire the data
    "FILTER",  # Name of the filter element used
    "PUPIL",  # Name of the pupil element used
    "GRATING", # Name of the grating element used
    "EXP_TYPE",  # Type of data in the exposure
    "CHANNEL",  # Instrument channel
    "SUBARRAY",  # Subarray used
    "NUMDTHPT",  # Total number of points in pattern
    "GS_RA",  # guide star right ascension
    "GS_DEC",  # guide star declination
    "CROWDFLD", # Are the FGSes in a crowded field?
    "GS_MAG", # guide star magnitude in FGS detector
]


SCIKEYS = [
    "RA_REF",
    "DEC_REF",
    "CRVAL1",
    "CRVAL2",
]


COLUMN_ORDER = {
    "IMAGE": [
        "instr",
        "detector",
        "visitype",
        "filter",
        "pupil",
        "channel",
        "subarray",
        "bkgdtarg",
        "nexposur",
        "numdthpt",
        "offset",
        "max_offset",
        "mean_offset",
        "sigma_offset",
        "err_offset",
        "sigma1_mean",
        "frac",
        "targ_frac",
    ],
    "SPEC": [
        "instr",
        "detector",
        "visitype",
        "filter",
        "grating",
        "subarray",
        "bkgdtarg",
        "is_imprt",
        "nexposur",
        "numdthpt",
        "targ_max_offset",
        "offset",
        "max_offset",
        "mean_offset",
        "sigma_offset",
        "err_offset",
        "sigma1_mean",
        "frac",
    ],
    "FGS": [
        "instr",
        "detector",
        "visitype",
        "subarray",
        "nexposur",
        "numdthpt",
        "crowdfld",
        "gs_mag",       
    ]
}

NORM_COLS = {
    "IMAGE": [
        "offset",
        "max_offset",
        "mean_offset",
        "sigma_offset",
        "err_offset",
        "sigma1_mean",
    ],
    "SPEC": [
        "targ_max_offset",     
        "offset",
        "max_offset",
        "mean_offset",
        "sigma_offset",
        "err_offset",
        "sigma1_mean",
    ],
    "FGS": ["gs_mag"],
}


KEYPAIR_DATA = {
    "instr": {"FGS": 0, "MIRI": 1, "NIRCAM": 2, "NIRISS": 3, "NIRSPEC": 4},
    "detector": {
        "NONE": 0,
        "GUIDER1": 1,
        "GUIDER1|GUIDER2": 2,
        "GUIDER2": 3,
        "MIRIFULONG": 4,
        "MIRIFULONG|MIRIFUSHORT": 5,
        "MIRIFULONG|MIRIFUSHORT|MIRIMAGE": 6,
        "MIRIFUSHORT": 7,
        "MIRIMAGE": 8,
        "NIS": 9,
        "NRCA1": 10,
        "NRCA1|NRCA2|NRCA3|NRCA4": 11,
        "NRCA1|NRCA2|NRCA3|NRCA4|NRCB1|NRCB2|NRCB3|NRCB4": 12,
        "NRCA1|NRCA3": 13,
        "NRCA1|NRCA3|NRCALONG": 14,
        "NRCA2": 15,
        "NRCA3": 16,
        "NRCA4": 17,
        "NRCALONG": 18,
        "NRCALONG|NRCBLONG": 19,
        "NRCB1": 20,
        "NRCB1|NRCB2|NRCB3|NRCB4": 21,
        "NRCB1|NRCBLONG": 22,
        "NRCB2": 23,
        "NRCB2|NRCB4": 24,
        "NRCB3": 25,
        "NRCB4": 26,
        "NRCBLONG": 27,
        "NRS1": 28,
        "NRS1|NRS2": 29,
        "NRS2": 30,
    },
    "filter": {
        "NONE": 0,
        "CLEAR": 1,
        "F070LP": 2,
        "F070W": 3,
        "F090W": 4,
        "F1000W": 5,
        "F100LP": 6,
        "F1065C": 7,
        "F110W": 8,
        "F1130W": 9,
        "F1140C": 10,
        "F115W": 11,
        "F1280W": 12,
        "F140M": 13,
        "F140X": 14,
        "F1500W": 15,
        "F150W": 16,
        "F150W2": 17,
        "F1550C": 18,
        "F170LP": 19,
        "F1800W": 20,
        "F182M": 21,
        "F187N": 22,
        "F200W": 23,
        "F2100W": 24,
        "F210M": 25,
        "F212N": 26,
        "F2300C": 27,
        "F250M": 28,
        "F2550W": 29,
        "F2550WR": 30,
        "F277W": 31,
        "F290LP": 32,
        "F300M": 33,
        "F322W2": 34,
        "F335M": 35,
        "F356W": 36,
        "F360M": 37,
        "F380M": 38,
        "F410M": 39,
        "F430M": 40,
        "F444W": 41,
        "F460M": 42,
        "F480M": 43,
        "F560W": 44,
        "F770W": 45,
        "FND": 46,
        "GR150C": 47,
        "GR150R": 48,
        "OPAQUE": 49,
        "P750L": 50,
        "WLP4": 51,
    },
    "pupil": {
        "NONE": 0,
        "CLEAR": 1,
        "CLEARP": 2,
        "F090W": 3,
        "F115W": 4,
        "F140M": 5,
        "F150W": 6,
        "F158M": 7,
        "F162M": 8,
        "F164N": 9,
        "F200W": 10,
        "F323N": 11,
        "F405N": 12,
        "F466N": 13,
        "F470N": 14,
        "FLAT": 15,
        "GDHS0": 16,
        "GDHS60": 17,
        "GR700XD": 18,
        "GRISMC": 19,
        "GRISMR": 20,
        "MASKBAR": 21,
        "MASKIPR": 22,
        "MASKRND": 23,
        "NRM": 24,
        "WLM8": 25,
        "WLP8": 26,
    },
    "grating": {
        "NONE": 0,
        "MIRROR": 1,
        "PRISM": 2,
        "G140M": 3,
        "G235M": 4,
        "G395M": 5,
        "G395H": 6,
        "G140H": 7,
        "G235H": 8,
    },
    "exp_type": {
        "NONE": 0,
        "FGS_FOCUS": 1,
        "FGS_IMAGE": 2,
        "FGS_INTFLAT": 3,
        "MIR_4QPM": 4,
        "MIR_FLATIMAGE": 5,
        "MIR_FLATIMAGE-EXT": 6,
        "MIR_FLATMRS": 7,
        "MIR_IMAGE": 8,
        "MIR_LRS-FIXEDSLIT": 9,
        "MIR_LRS-FIXEDSLIT|NIS_SOSS": 10,
        "MIR_LRS-SLITLESS": 11,
        "MIR_LYOT": 12,
        "MIR_MRS": 13,
        "NIS_AMI": 14,
        "NIS_DARK": 15,
        "NIS_EXTCAL": 16,
        "NIS_IMAGE": 17,
        "NIS_LAMP": 18,
        "NIS_SOSS": 19,
        "NIS_WFSS": 20,
        "NRC_CORON": 21,
        "NRC_DARK": 22,
        "NRC_GRISM": 23,
        "NRC_IMAGE": 24,
        "NRC_LED": 25,
        "NRC_TSGRISM": 26,
        "NRC_TSIMAGE": 27,
        "NRC_WFSS": 28,
        "NRC_WFSS|NRS_AUTOFLAT|NRS_AUTOWAVE": 29,
        "NRS_AUTOFLAT": 30,
        "NRS_AUTOFLAT|NRS_AUTOWAVE|NRS_FIXEDSLIT": 31,
        "NRS_AUTOWAVE": 32,
        "NRS_AUTOWAVE|NRS_IFU": 33,
        "NRS_BRIGHTOBJ": 34,
        "NRS_FIXEDSLIT": 35,
        "NRS_IFU": 36,
        "NRS_LAMP": 37,
        "NRS_MIMF": 38,
        "NRS_MSASPEC": 39,
    },
    "channel": {"NONE": 0, "12": 1, "34": 2, "LONG": 3, "SHORT": 4},
    "subarray": {
        "NONE": 0,
        "ALLSLITS": 1,
        "BRIGHTSKY": 2,
        "FULL": 3,
        "MASK1065": 4,
        "MASK1140": 4,
        "MASK1550": 4,
        "MASKLYOT": 4,
        "SLITLESSPRISM": 5,
        "SUB128": 6,
        "SUB160": 6,
        "SUB160P": 6,
        "SUB2048": 6,
        "SUB256": 6,
        "SUB32": 6,
        "SUB320": 6,
        "SUB320A335R": 6,
        "SUB320A430R": 6,
        "SUB320ALWB": 6,
        "SUB32TATS": 6,
        "SUB32TATSGRISM": 6,
        "SUB400P": 6,
        "SUB512": 6,
        "SUB512S": 6,
        "SUB64": 6,
        "SUB640": 6,
        "SUB640A210R": 6,
        "SUB640ASWB": 6,
        "SUB64FP1A": 6,
        "SUB64P": 6,
        "SUB80": 6,
        "SUB96DHSPILA": 6,
        "SUBAMPCAL": 6,
        "SUBFSA210R": 6,
        "SUBFSA335R": 6,
        "SUBFSA430R": 6,
        "SUBFSALWB": 6,
        "SUBFSASWB": 6,
        "SUBGRISM128": 6,
        "SUBGRISM256": 6,
        "SUBGRISM64": 6,
        "SUBNDA210R": 6,
        "SUBNDA335R": 6,
        "SUBNDA430R": 6,
        "SUBNDALWBL": 6,
        "SUBNDALWBS": 6,
        "SUBNDASWBS": 6,
        "SUBS200A1": 6,
        "SUBS200A2": 6,
        "SUBS400A1": 6,
        "SUBSTRIP256": 6,
        "SUBSTRIP96": 6,
        "SUBTAAMI": 6,
        "SUBTASOSS": 6,
        "WFSS128C": 7,
        "WFSS128R": 7,
        "WFSS64C": 7,
        "WFSS64R": 7,
    },
    "visitype": {
        "NONE": 0,
        ".+WFSC.+": 1,
        "PARALLEL_PURE": 2,
        "PARALLEL_SLEW_CALIBRATION": 3,
        "PRIME_TARGETED_FIXED": 4,
        "PRIME_TARGETED_MOVING": 5,
        "PRIME_UNTARGETED": 6,
        "PRIME_WFSC_ROUTINE": 7,
        "PRIME_WFSC_SENSING_CONTROL": 8,
        "PRIME_WFSC_SENSING_ONLY": 9,
    },
}
