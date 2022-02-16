"""
Retrieve dataset archive (.zip) files from the web, an s3 bucket, or local disk.
This script is primarily intended as a retrieval and extraction step before launching spacekit.dashboard
For more customized control of dataset retrieval (such as your own), use the spacekit.extractor.scrape module.

Examples:
"datasets": if set, chooses specific archive dataset filenames to retrieve and extract

src: "git" -> Fetch and extract from one of the spacekit data archives:
archive: name of collection (see ``spacekit.datasets.meta``)
download(scrape="git:calcloud")
download(scrape="git:svm")

src: "file" -> Fetch and extract from path on local disk
archive: path
download(scrape="file:another/path/to/data)

src: "s3" -> Fetch and extract from an S3 bucket on AWS
archive: bucketname
"datasets" -> string of S3 archive file basenames separated by comma (without the .zip or .tgz suffix)
download(scrape="s3:mybucketname", "2021-11-04-1636048291,2021-10-28-1635457222,2021-08-22-1629663047")

# TODO
src: "web" -> Fetch and extract from web (experimental)
archive: json filepath containing metadata structured similar to dictionaries in ``spacekit.datasets.meta``
"""
import argparse
import sys
import json
from spacekit.datasets.meta import spacekit_collections
from spacekit.extractor.scrape import WebScraper, S3Scraper, FileScraper

# from spacekit.datasets import scrape_archives, import_collection

DATA = "spacekit.datasets.data"


def download(scrape="file:data", datasets=None, dest="."):
    src, archive = scrape.split(":")
    if src == "git":
        print("Scraping Github Archive")
        cc = spacekit_collections[archive]  # "calcloud", "svm"
        scraper = WebScraper(cc["uri"], cc["data"], cache_dir=dest)
    elif src == "s3":
        print("Scraping S3")
        scraper = S3Scraper(archive, pfx="archive", cache_dir=dest)
        fnames = datasets.split(",")
        scraper.make_s3_keys(fnames=fnames)
    elif src == "file":  # args.src == "file"
        print("Scraping local directory")
        p = [f"{archive}/*.zip", f"{archive}/*"]
        scraper = FileScraper(patterns=p, clean=False, cache_dir=dest)
    elif src == "web":
        with open(archive, "r") as j:
            collection = json.load(j)
        scraper = WebScraper(collection["uri"], collection["data"], cache_dir=dest)
    try:
        scraper.scrape()
        if scraper.fpaths:
            print("Datasets: ", scraper.fpaths)
            return scraper.fpaths
    except Exception as e:
        print("Could not locate datasets.")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scrape", default="git:calcloud")
    parser.add_argument(
        "-d",
        "--datasets",
        default="2021-11-04-1636048291,2021-10-28-1635457222,2021-08-22-1629663047",
    )
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    fpaths = download(scrape=args.scrape, datasets=args.datasets, dest=args.out)
    # fpaths = import_collection(archive)
    # scrape_archives(spacekit_collections[archive], data_home=DD)
