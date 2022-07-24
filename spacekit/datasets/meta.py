calcloud = {
    # "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/rev-0.3.2/spacekit/datasets/data/calcloud",
    "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/spacekit/datasets/data/calcloud",
    "data": {
        "2022-02-14": {
            "fname": "2022-02-14-1644848448.zip",
            "hash": "d39482d148736f2b2e8ad1ca19f0d6797bab6582700f174df0344fc9f91687dd",
        },
        "2021-11-04": {
            "fname": "2021-11-04-1636048291.zip",
            "hash": "d0cb8c35554527d8dda96f259952b0d1436f6fda5f4d11ced73d7e88f328aff4",
        },
        "2021-10-28": {
            "fname": "2021-10-28-1635457222.zip",
            "hash": "dce221998c366486e3412b13ad90493bad6e234d54e0849a635700cb8494acb4",
        },
        "2021-08-22": {
            "fname": "2021-08-22-1629663047.zip",
            "hash": "d8f17819282add50c3c27fc8eada314ffad72a60c8b0e8639d98b5f6f77602bb",
        },
    },
    "model": {
        "basepath": "spacekit.builder.trained_networks",
        "fname": "calmodels.zip",
        "hash": "b36e310874f7cd2c50d4e5c04438118af7751c69",
    },
}

svm = {
    "uri": "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/spacekit/datasets/data/svm",
    "data": {
        "2022-02-14": {
            "fname": "2022-02-14-1644850390.zip",
            "hash": "7f5ad34c2265219dd62eeb9a79612e69b2b1daf36735d2f9803898cfd1944dd7",
            "desc": "ensembleSVM training results",
        },
        "2022-01-30": {
            "fname": "2022-01-30-1643523529.zip",
            "hash": "5f1dde6e3177533e2c5e7b0ed81bce8daaccb68970e1f0d6614590674a0195c6",
            "desc": "ensembleSVM training results with additional synthetic data",
        },
        "2022-01-16": {
            "fname": "2022-01-16-1642337739.zip",
            "hash": "6a52e1aaf8eb4949bd906d70b45c8289e9eaa045c334d47a411434cf134a7bfd",
            "desc": "ensembleSVM training results 1-16-2022",
        },
        "2021-07-28": {
            "fname": "svm_labeled_2021-07-28.csv",
            "hash": "b7ac1163e63ad619a74fb1c27ea2ea3b5f8dd0bc44b0071214c7210ffd95cd9e",
            "desc": "labeled training set with synthetic data",
        },
        "2021-10-06": {
            "fname": "svm_unlabeled_2021-10-06.csv",
            "hash": "c0bc2c6baa7e130614aa2f75cb1b71e048e117ccf2cad8b2d2cd6364b1248f2f",
            "desc": "unlabeled test set created 10-6-2021",
        },
        "2021-11-02": {
            "fname": "2021-11-02_predictions.zip",
            "hash": "b0bf232313460692a3d00ce2b73fae7627d3c1b7c4f9762e1ff7cdc5e9d2e5d0",
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
