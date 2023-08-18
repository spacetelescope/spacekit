from spacekit.builder.architect import (
    MemoryClassifier,
    MemoryRegressor,
    WallclockRegressor,
)
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from spacekit.skopes.hst.cal.config import COLUMN_ORDER
from spacekit.extractor.scrape import S3Scraper
from spacekit.extractor.load import save_multitype_data, zip_subdirs
from spacekit.logger.log import SPACEKIT_LOG

try:
    from scikeras.wrappers import KerasClassifier, KerasRegressor
except ImportError:
    KerasClassifier, KerasRegressor = None, None


def check_scikeras():
    return KerasClassifier is not None


def k_estimator(buildClass, n_splits=10, y=None, stratify=False):
    bld = buildClass()
    bld.build()

    if stratify is True:
        estimator = KerasClassifier(
            model=bld.model,
            epochs=bld.epochs,
            batch_size=bld.batch_size,
            verbose=bld.verbose,
        )
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    else:
        estimator = KerasRegressor(
            build_fn=bld.model,
            epochs=bld.epochs,
            batch_size=bld.batch_size,
            verbose=bld.verbose,
        )
        kfold = KFold(n_splits=n_splits, shuffle=True)
    if y:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        return estimator, kfold, y
    else:
        return estimator, kfold


def kfold_cross_val(data, target_col, s3=None, data_path=None, verbose=2, n_jobs=-2):
    # evaluate using 10-fold cross validation
    X = data.data[COLUMN_ORDER["asn"]]
    y = data.data[target_col]

    # run estimator
    if target_col == "mem_bin":
        estimator, kfold, y = k_estimator(MemoryClassifier, y=y, stratify=True)

    elif target_col == "memory":
        estimator, kfold = k_estimator(MemoryRegressor)

    elif target_col == "wallclock":
        estimator, kfold = k_estimator(WallclockRegressor)

    print("\nStarting KFOLD Cross-Validation...")
    start = time.time()
    results = cross_val_score(estimator, X, y, cv=kfold, n_jobs=n_jobs, verbose=verbose)
    end = time.time()

    duration = end - start

    if target_col == "mem_bin":
        score = np.mean(results)
    else:
        score = np.sqrt(np.abs(np.mean(results)))
    print(f"\nKFOLD scores: {results}\n")
    print(f"\nMean Score: {score}\n")

    kfold_dict = {"kfold": {"results": list(results), "score": score, "time": duration}}

    save_multitype_data(kfold_dict, target_col)
    zip_subdirs(target_col, zipname="kfold.zip")
    if s3 is not None:
        prefix = "training" if data_path is None else data_path
        S3Scraper.s3_upload(["kfold.zip"], s3, f"{prefix}/results/{target_col}")
    return kfold_dict


def run_kfold(data, s3=None, data_path=None, models=[], n_jobs=-2):
    """_summary_

    Parameters
    ----------
    data : object
        spacekit.preprocessor.prep.HstCalPrep data object
    s3 : str, optional
        S3 bucket name, by default None
    data_path : str, optional
        path to local dataset directory, by default None
    models : list, optional
        saved model names, by default []
    n_jobs : int, optional
        kfold njobs, by default -2
    """
    if not check_scikeras():
        SPACEKIT_LOG.error("scikeras not installed.")
        raise ImportError(
            "You must install scikeras (`pip install scikeras`) "
            "for the validation module to work."
            "\n\nInstall extra deps via `pip install spacekit[x]`"
        )
    for target in models:
        kfold_cross_val(data, target, s3=s3, data_path=data_path, n_jobs=n_jobs)
