calcloud = {
    "uri": "https://zenodo.org/record/7830049/files",
    "data": {
        "2022-02-14": {
            "fname": "hst_cal_std_2022-02-14.zip?download=1",
            "hash": "7522ec6bdaffaa06827e9ec9781b5182",
            "desc": "data, model and training results",
            "key": "2022-02-14-1644848448",
            "size": "12.9MB"
        },
        "2021-11-04": {
            "fname": "hst_cal_std_2021-11-04.zip?download=1",
            "hash": "d1ba4329ee18219e1a562d7a72c96368",
            "desc": "data, model and training results",
            "key": "2021-11-04-1636048291",
            "size": "11.6MB"
        },
        "2021-10-28": {
            "fname": "hst_cal_std_2021-10-28.zip?download=1",
            "hash": "09fe043a61165def660ca0209a4c562e",
            "desc": "data, model and training results",
            "key": "2021-10-28-1635457222",
            "size": "8.6MB"
        },
        "2021-08-22": {
            "fname": "hst_cal_std_2021-08-22.zip?download=1",
            "hash": "2ac4857954ddcd384c43a9a0dd8d7cff",
            "desc": "data, model and training results",
            "key": "2021-08-22-1629663047",
            "size": "6MB"
        },
    },
    "model": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "calmodels.zip",
        "hash": "000378942a1dcb662590ac6d911b8ba1e59d54248254d23e03427bb048597298",
        "size": "2MB"
    },
}

svm = {
    "uri": "https://zenodo.org/record/7830049/files",
    "data": {
        "2022-02-14": {
            "fname": "hst_drz_svm_2022-02-14.zip?download=1",
            "hash": "c0590ee3277aa1f2607a0d9ad827db27",
            "desc": "latest model, data and training results",
            "key": "2022-02-14-1644850390",
            "size": "18MB"
        },
        "2022-01-30": {
            "fname": "hst_drz_svm_2022-01-30.zip?download=1",
            "hash": "758334cf0f4e80038e2f966ac5c044a1",
            "desc": "retrained model, data and training results",
            "key": "2022-01-30-1643523529",
            "size": "18MB"
        },
        "2022-01-16": {
            "fname": "hst_drz_svm_2022-01-16.zip?download=1",
            "hash": "8c0e87943afeb62404df0d2e529051d6",
            "desc": "baseline model, data and training results",
            "key": "2022-01-16-1642337739",
            "size": "34.1kB"
        },
        "2021-07-28": {
            "fname": "svm_labeled_2021-07-28.csv?download=1",
            "hash": "29fe3fde4e0f826e8c6d4b85a21a3690",
            "desc": "labeled test set created 07-28-2021 for drizzlepac 3.3.1",
            "key": "svm_labeled_2021-07-28.csv",
            "size": "179.1kB"
        },
        "2021-10-06": {
            "fname": "svm_unlabeled_2021-10-06.csv?download=1",
            "hash": "2979fd5b0e24663ce0ae8ff93ebd1ca1",
            "desc": "unlabeled test set created 10-6-2021 for drizzlepac 3.3.1",
            "key": "svm_unlabeled_2021-10-06.csv",
            "size": "102kB"
        }
    },
    "model": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "ensemble.zip",
        "hash": "9ef2b5ddd078544c98a19e2c26fcccb34ae36c742b86ee3b9434ceab6270c07d",
        "size": "17MB"
    },
}

k2 = {
    "uri": "https://zenodo.org/record/7830049/files",
    "data": {
        "test": {
            "fname": "k2-exo-flux-ts-test.csv.zip?download=1",
            "hash": "02646070957656c38c6c9dff66552684",
            "desc": "k2 flux time series test set",
            "key": "exoTest.csv",
            "size": "5.8MB"
        },
        "train": {
            "fname": "k2-exo-flux-ts-train.csv.zip?download=1",
            "hash": "f26bcc2dbc30544678e0c15c1a67fe6b",
            "desc": "k2 flux time series training set",
            "key": "exoTrain.csv",
            "size": "51.7MB"
        },
    },
}

spacekit_collections = {"calcloud": calcloud, "svm": svm, "k2": k2}

networks = {
    "calcloud": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "calmodels.zip",
        "hash": "000378942a1dcb662590ac6d911b8ba1e59d54248254d23e03427bb048597298",
        "size": "2MB"
    },
    "svm": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "ensemble.zip",
        "hash": "9ef2b5ddd078544c98a19e2c26fcccb34ae36c742b86ee3b9434ceab6270c07d",
        "size": "17MB"
    },
}
