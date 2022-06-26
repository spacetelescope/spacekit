"""
Base IO code for all datasets (borrowing concepts from sklearn.datasets and keras.utils.load_data)
"""
from importlib import resources
from spacekit.extractor.scrape import WebScraper, FileScraper
from spacekit.analyzer.scan import import_dataset, CalScanner, SvmScanner
from spacekit.datasets.meta import spacekit_collections

DATA = "spacekit.datasets.data"


def import_collection(name, date_key=None, data_home=None):
    source = f"{DATA}.{name}"
    archives = spacekit_collections[name]["data"]
    if date_key is None:
        # fetch 3 most recent
        fnames = [archives[date]["fname"] for date in archives.keys()][:3]
    elif type(date_key) == list:
        fnames = [archives[date]["fname"] for date in date_key]
    else:
        fnames = [archives[date_key]["fname"]]
    scr = FileScraper(cache_dir=data_home, clean=False)
    for fname in fnames:
        with resources.path(source, fname) as archive:
            scr.fpaths.append(archive)
    fpaths = scr.extract_archives()
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
    df = load_from_archive(cal, fpath=fpath, date_key=date_key, scanner=CalScanner)
    return df


def load_svm(fpath=None, date_key=None):
    svm = spacekit_collections["svm"]
    df = load_from_archive(svm, fpath=fpath, date_key=date_key, scanner=SvmScanner)
    return df


def load_k2():
    k2 = spacekit_collections["k2"]
    train, test = scrape_archives(k2)
    return train, test


def load(name="calcloud", date_key=None, fpath=None, data_home=None):
    if fpath is None:
        fpath = import_collection(name, date_key=date_key, data_home=data_home)
    if name == "calcloud":
        scn = CalScanner(perimeter=fpath)
    elif name == "svm":
        scn = SvmScanner(perimeter=fpath)
    df = scn.load_dataframe(kwargs=scn.kwargs, decoder=scn.decoder)
    return df
