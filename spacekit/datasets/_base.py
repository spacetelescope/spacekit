"""
Base IO code for all datasets (borrowing concepts from sklearn.datasets and keras.utils.load_data)
"""
from spacekit.extractor.scrape import WebScraper
from spacekit.analyzer.scan import (
    import_dataset,
    HstCalScanner,
    HstSvmScanner,
    JwstCalScanner,
)
from spacekit.datasets.meta import spacekit_collections


def import_collection(name, date_key=None, data_home=None):
    archives = spacekit_collections[name]
    datasets = {}
    if date_key is None:
        date_key = list(archives["data"].keys()[:3])
    elif isinstance(date_key, str):
        date_key = [date_key]
    for d in date_key:
        datasets[d] = archives["data"][d]
    fpaths = WebScraper(archives["uri"], datasets, cache_dir=data_home).scrape()
    return fpaths


def scrape_archives(archives, data_home=None):
    """Download zip archives of training data iterations (includes datasets, models, and results).

    Returns
    -------
    list
        list of paths to retrieved and extracted dataset collection
    """
    fpaths = WebScraper(archives["uri"], archives["data"], cache_dir=data_home).scrape()
    return fpaths


def download_single_archive(archives, date_key=None, data_home=None):
    uri = archives["uri"]
    data = archives["data"]
    if date_key is None:
        # default to most recent
        date_key = sorted(list(data.keys()))[-1]
    # limit data download to single archive
    dataset = {date_key: data[date_key]}
    scraper = WebScraper(uri, dataset, cache_dir=data_home).scrape()
    fpath = scraper.fpaths[0]
    print(fpath)
    return fpath


def load_from_archive(
    archives, fpath=None, date_key=None, scanner=None, data_home=None
):
    if fpath is None:
        fpath = download_single_archive(
            archives, date_key=date_key, data_home=data_home
        )
    if scanner:
        scn = scanner(perimeter=fpath)
        df = scn.load_dataframe(kwargs=scn.kwargs, decoder=scn.decoder)
    else:
        df = import_dataset(filename=fpath)
    return df


def load_cal(fpath=None, date_key=None):
    cal = spacekit_collections["calcloud"]
    df = load_from_archive(cal, fpath=fpath, date_key=date_key, scanner=HstCalScanner)
    return df


def load_svm(fpath=None, date_key=None):
    svm = spacekit_collections["svm"]
    df = load_from_archive(svm, fpath=fpath, date_key=date_key, scanner=HstSvmScanner)
    return df


def load_k2():
    k2 = spacekit_collections["k2"]
    train, test = scrape_archives(k2)
    return train, test


def load_jwst_cal(fpath=None, date_key=None):
    jw = spacekit_collections["jwst_cal"]
    df = load_from_archive(jw, fpath=fpath, date_key=date_key, scanner=JwstCalScanner)
    return df


def load(name="calcloud", date_key=None, fpath=None, data_home=None):
    if fpath is None:
        fpath = import_collection(name, date_key=date_key, data_home=data_home)
    if name == "calcloud":
        scn = HstCalScanner(perimeter=fpath)
    elif name == "svm":
        scn = HstSvmScanner(perimeter=fpath)
    elif name == "jwst_cal":
        scn = JwstCalScanner(perimeter=fpath)
    scn.load_dataframe()
    return scn.df
