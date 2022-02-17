calcloud = {
    # "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/spacekit/datasets/hst/calcloud",
    "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/spacekit/datasets/data/calcloud",
    "data": {
        "2021-11-04": {
            "fname": "2021-11-04-1636048291.zip",
            "hash": "53bbf7486c4754b60b2f8ff3898ddb3bb6f744c9",
            # "hash": "7d9827bcf94cfbd0d235ea2ff8b4fcb519fd5ff215c286d03dd5ecb1322d0518",
        },
        "2021-10-28": {
            "fname": "2021-10-28-1635457222.zip",
            "hash": "96222742bee071bfa32a403720e6ae8a53e66f56",
        },
        "2021-08-22": {
            "fname": "2021-08-22-1629663047.zip",
            "hash": "ef872adc24ae172d9ccc8a74565ad81104dee2c0",
        },
    },
    "model": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "calmodels.zip",
        "hash": "b36e310874f7cd2c50d4e5c04438118af7751c69",
    },
}

svm = {
    # "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/spacekit/datasets/hst/svm/",
    "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/spacekit/datasets/data/svm",
    "data": {
        "2022-01-16": {
            "fname": "2022-01-16-1642337739.zip",
            "hash": "2c9d2a2ae9375c53641ff03050a86d1ba7fd5ecc",
            "desc": "ensembleSVM training results 1-16-2021",
        },
        "2021-01-30": {
            "fname": "2022-01-30-1643523529.zip",
            "hash": "d76ca02bac5b7ea97103f70674d2d8803fa7b903",
            # "d4ec418ddbf44401975a275a53b69447fa405c13795a68f47014b96435bf4293"
            "desc": "ensembleSVM training results 1-30-2021 additional synthetic data",
        },
        "2021-07-28": {
            "fname": "svm_2021-07-28.csv",
            "hash": "575e5d9c928279917e225f66adbc44f7e31824bb",
            "desc": "original labeled training dataframe",
        },
        "2021-10-06": {
            "fname": "svm_2021-10-06.csv",
            "hash": "cca34022ca896d35b9ebbbd502b1dc22d0df34a1",
            "desc": "unlabeled regression dataframe 10-6-2021",
        },
        "2021-11-02": {
            "fname": "2021-11-02_predictions.zip",
            "hash": "408620cb25cc666a8123416205e193d507487627",
            "desc": "alignment predictions for 11-02 unlabeled data",
        },
    },
    "model": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "ensembleSVM.zip",
        "hash": "15a88d6a14e018b68a6c411fccd5ffe496fcb23d",
    },
}

k2 = {
    "uri": "https://github.com/alphasentaurii/starskope/raw/master/data",
    "data": {
        "train": {
            "fname": "exoTrain.csv.zip",
            "hash": "7160efee11cb1eaec5f8c5fba3b41357b4e015f5",
        },
        "test": {
            "fname": "exoTest.csv.zip",
            "hash": "342182de43ae2b686f2d74946700e1a42075f101",
        },
    },
}

spacekit_collections = {"calcloud": calcloud, "svm": svm, "k2": k2}

networks = {
    "calcloud": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "calmodels.zip",
        "hash": "b36e310874f7cd2c50d4e5c04438118af7751c69",
    },
    "svm": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "ensembleSVM.zip",
        "hash": "602b9d9129940820e57ed3db87cb70eb15aec503",
    },
}
