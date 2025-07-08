ZID = "15839291" # zenodo DOI record

calcloud = {
    "uri": f"https://zenodo.org/record/{ZID}/files",
    "data": {
        "2022-02-14": {
            "fname": "hst_cal_std_2022-02-14.zip?download=1",
            "hash": "3ca674b0d81e98e319f6e80d78ce8db8",
            "desc": "data, model and training results",
            "key": "2022-02-14-1644848448",
            "size": "9.7MB",
        },
        "2021-11-04": {
            "fname": "hst_cal_std_2021-11-04.zip?download=1",
            "hash": "029fcba4e4f576faf259035b2c20d13c",
            "desc": "data, model and training results",
            "key": "2021-11-04-1636048291",
            "size": "11.4MB",
        },
        "2021-10-28": {
            "fname": "hst_cal_std_2021-10-28.zip?download=1",
            "hash": "2dc85c941627c8aec7bac0ebe725d7dd",
            "desc": "data, model and training results",
            "key": "2021-10-28-1635457222",
            "size": "8.5MB",
        },
        "2021-08-22": {
            "fname": "hst_cal_std_2021-08-22.zip?download=1",
            "hash": "6a422aa2913194ff2af03398ca2403be",
            "desc": "data, model and training results",
            "key": "2021-08-22-1629663047",
            "size": "5.9MB",
        },
    },
    "model": {
        "fname": "hst_cal.zip",
        "hash": "acb07516b9385ce7182577ef7216ae75",
        "desc": "hst calcloud resource prediction models",
        "key": "hst_cal",
        "size": "385.47kB",
    },
}

svm = {
    "uri": f"https://zenodo.org/record/{ZID}/files",
    "data": {
        "2022-02-14": {
            "fname": "hst_drz_svm_2022-02-14.zip?download=1",
            "hash": "2660e4457b19ce01b625d7a668d04e89",
            "desc": "latest model, data and training results",
            "key": "2022-02-14-1644850390",
            "size": "5MB",
        },
        "2022-01-30": {
            "fname": "hst_drz_svm_2022-01-30.zip?download=1",
            "hash": "370fdbdb8d2b45b2ded998dbcc6927a5",
            "desc": "updated model, data and training results",
            "key": "2022-01-30-1643523529",
            "size": "5MB",
        },
        "2022-01-16": {
            "fname": "hst_drz_svm_2022-01-16.zip?download=1",
            "hash": "7a5f99b1448ba09e7323f48fd7899ae5",
            "desc": "baseline model training results",
            "key": "2022-01-16-1642337739",
            "size": "24.8kB",
        },
        "2021-07-28": {
            "fname": "svm_labeled_2021-07-28.csv?download=1",
            "hash": "29fe3fde4e0f826e8c6d4b85a21a3690",
            "desc": "labeled test set created 07-28-2021 for drizzlepac 3.3.1",
            "key": "svm_labeled_2021-07-28.csv",
            "size": "179.1kB",
        },
        "2021-10-06": {
            "fname": "svm_unlabeled_2021-10-06.csv?download=1",
            "hash": "2979fd5b0e24663ce0ae8ff93ebd1ca1",
            "desc": "unlabeled test set created 10-6-2021 for drizzlepac 3.3.1",
            "key": "svm_unlabeled_2021-10-06.csv",
            "size": "102kB",
        },
    },
    "model": {
        "fname": "svm_align.zip",
        "hash": "c98492b1e729407c72a69d2c739232be",
        "desc": "hst svm alignment prediction models",
        "key": "svm_align",
        "size": "4.92MB",
    },
}


jwst_cal = {
    "uri": f"https://zenodo.org/record/{ZID}/files",
    "data": {
        "2023-08-02": {
            "fname": "",
            "hash": "",
            "desc": "",
            "key": "",
            "size": "",
        },
        "2024-03-28": {
            "fname": "",
            "hash": "",
            "desc": "",
            "key": "",
            "size": "",
        },
        "2025-02-25": {
            "fname": "",
            "hash": "",
            "desc": "",
            "key": "",
            "size": "",
        },
        "2025-06-23": {
            "fname": "",
            "hash": "",
            "desc": "",
            "key": "",
            "size": "",
        },
    },
    "model": {
        "fname": "jwst_cal.zip",
        "hash": "c923e4842c0d61c0c06ce566361a1091",
        "desc": "JWST Calibration Processing Resource Prediction Models",
        "key": "jwst_cal",
        "size": "1.5MB",
    },
}


k2 = {
    "uri": f"https://zenodo.org/record/{ZID}/files",
    "data": {
        "test": {
            "fname": "k2-exo-flux-ts-test.csv.zip?download=1",
            "hash": "02646070957656c38c6c9dff66552684",
            "desc": "k2 flux time series test set",
            "key": "exoTest.csv",
            "size": "5.8MB",
        },
        "train": {
            "fname": "k2-exo-flux-ts-train.csv.zip?download=1",
            "hash": "f26bcc2dbc30544678e0c15c1a67fe6b",
            "desc": "k2 flux time series training set",
            "key": "exoTrain.csv",
            "size": "51.7MB",
        },
    },
}


spacekit_collections = {
    "calcloud": calcloud,
    "svm": svm,
    "jwst_cal": jwst_cal,
    "k2": k2,
}

# Package imports
networks = {
    "calcloud": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "hst_cal.zip",
        "hash": "3dcfd4760154dee890f7943e237e679a43eef82d3c0838ff62ecf17e4c3702fb",
        "size": "385.47kB",
    },
    "svm": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "svm_align.zip",
        "hash": "cda062ba768374802e4e8b919c0dcce79ce3b72e29004cc09601a67da454b008",
        "size": "4.92MB",
    },
    "jwst_cal": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "jwst_cal.zip",
        "hash": "79f62531ec6dcb9958816bda4ba51b10e326b2e907a47898f5e48a063c139358",
        "size": "1.5MB",
    },
}
