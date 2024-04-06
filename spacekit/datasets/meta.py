calcloud = {
    "uri": "https://zenodo.org/record/10895592/files",
    "data": {
        "2022-02-14": {
            "fname": "hst_cal_std_2022-02-14.zip?download=1",
            "hash": "e9375c65feb413a660cb737a9a8a3a73",
            "desc": "data, model and training results",
            "key": "2022-02-14-1644848448",
            "size": "11.4MB",
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
        "hash": "ffe7d00504ad5d6ebc79488abe612474",
        "desc": "hst calcloud resource prediction models",
        "key": "hst_cal",
        "size": "385.42kB",
    },
}

svm = {
    "uri": "https://zenodo.org/record/10895592/files",
    "data": {
        "2022-02-14": {
            "fname": "hst_drz_svm_2022-02-14.zip?download=1",
            "hash": "3d6ade5510cc7aad4c2c45329c0751fe",
            "desc": "latest model, data and training results",
            "key": "2022-02-14-1644850390",
            "size": "18MB",
        },
        "2022-01-30": {
            "fname": "hst_drz_svm_2022-01-30.zip?download=1",
            "hash": "9ad9b95c20b82877f1ea7b0fa64b10a1",
            "desc": "updated model, data and training results",
            "key": "2022-01-30-1643523529",
            "size": "18MB",
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
        "hash": "af3b019cc3079cb29fd01bae58e9560d",
        "desc": "hst svm alignment prediction models",
        "key": "svm_align",
        "size": "4.92MB",
    },
}


jwst_cal = {
    "uri": "https://zenodo.org/record/10895592/files",
    "data": {
        "2023-08-02": {
            "fname": "",
            "hash": "",
            "desc": "",
            "key": "",
            "size": "",
        },
    },
    "model": {
        "fname": "jwst_cal.zip",
        "hash": "f3b6dbcc3fc92a6f806f5b5464e68c02",
        "desc": "JWST Calibration Processing Resource Prediction Models",
        "key": "jwst_cal",
        "size": "1.5MB",
    },
}


k2 = {
    "uri": "https://zenodo.org/record/10895592/files",
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
        "hash": "f3422c98d93520898e10d068a1e434dd5caf40e632760687df377c1b2bccff57",
        "size": "385.42kB",
    },
    "svm": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "svm_align.zip",
        "hash": "577a87fab157fdadc657af0dcd67b55af2d89d855d5597bd96671b20ff135636",
        "size": "4.92MB",
    },
    "jwst_cal": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "jwst_cal.zip",
        "hash": "b61d0281fd149e2b0ffcded81ba54b24b479c5b7191072be2bfb59464d779187",
        "size": "1.5MB",
    },
}
