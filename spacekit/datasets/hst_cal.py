from spacekit.extractor.scrape import WebScraper
from spacekit.analyzer.scan import import_dataset

calcloud_uri = "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/datasets/hst/calcloud/"

calcloud_data = {
    "2021-11-04": {
        "fname": "2021-11-04-1636048291.zip",
        "hash": "53bbf7486c4754b60b2f8ff3898ddb3bb6f744c9",
    },
    "2021-10-28": {
        "fname": "2021-10-28-1635457222.zip",
        "hash": "96222742bee071bfa32a403720e6ae8a53e66f56",
    },
    "2021-08-22": {
        "fname": "2021-08-22-1629663047.zip",
        "hash": "ef872adc24ae172d9ccc8a74565ad81104dee2c0",
    },
}


def download_single_archive(date_key=None):
    if date_key is None:
        # default to most recent
        date_key = sorted(list(calcloud_data.keys()))[-1]
    # limit data download to single archive
    dataset = {date_key: calcloud_data[date_key]}
    scraper = WebScraper(calcloud_uri, dataset).scrape_repo()
    fpath = scraper.fpaths[0]
    print(fpath)
    return fpath


def download_archives():
    """Download zip archives of calcloud hst training data iterations (including datasets, models, and results).

    Returns
    -------
    list
        list of paths to extracted dataset archives
    """
    fpaths = WebScraper(calcloud_uri, calcloud_data).scrape_repo()
    return fpaths


def load_data(fpath=None, date_key=None):
    if fpath is None:
        fpath = download_single_archive(date_key=date_key)
    df = import_dataset(
        filename=fpath,
        kwargs=dict(index_col="ipst"),
        decoder_key={"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}},
    )
    return df
