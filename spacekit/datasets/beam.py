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
import os
import json
from spacekit.datasets.meta import spacekit_collections
from spacekit.extractor.scrape import WebScraper, S3Scraper, FileScraper


DATA = "spacekit.datasets.data"
S3PREFIX = os.environ.get("PFX", "archive")


def download(scrape="file:data", datasets="2022-02-14,2021-11-04,2021-10-28", dest="."):
    src, archive = scrape.split(":")
    datasets = datasets.split(",")
    if src in ["pkg", "git", "web"]:
        if archive.split(".")[-1] == "json":
            print("Scraping web via custom json file")
            with open(archive, "r") as j:
                collection = json.load(j)
                scraper = WebScraper(collection["uri"], collection["data"], cache_dir=dest)
        elif archive in spacekit_collections.keys():
            print("Scraping Spacekit Collection")
            cc = spacekit_collections[archive]  # "calcloud", "svm"
            dd = {}
            for d in datasets:
                dd[d] = cc["data"][d]
            scraper = WebScraper(cc["uri"], dd, cache_dir=dest)
        else:
            print(f"Error: point to a custom json file or one of the spacekit collections: {list(spacekit_collections.keys())}")
    elif src == "s3":
        print("Scraping S3")
        scraper = S3Scraper(archive, pfx=S3PREFIX, cache_dir=dest)
        scraper.make_s3_keys(fnames=datasets)
    elif src == "file":
        print("Scraping local directory")
        p = [f"{archive}/*.zip", f"{archive}/*"]
        scraper = FileScraper(patterns=p, clean=False, cache_dir=dest)
    if scraper:
        try:
            scraper.scrape()
            if scraper.fpaths:
                print("Datasets: ", scraper.fpaths)
                return scraper.fpaths
        except Exception as e:
            print("Could not locate datasets.")
            print(e)
            sys.exit(1)
    elif fpaths:
        return fpaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scrape",
        default="web:calcloud",
        help="Uses a key:uri format where options for the key are limited to pkg, s3, file, or git and the uri could be your own custom location if not using the default datasets.  Examples are web:calcloud, web:custom.json, s3:mybucket, file:myfolder. Visit spacekit.readthedocs.io for more info.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        default="2022-02-14,2021-11-04,2021-10-28",
        help="Comma-separated string of keys identifying each dataset",
    )
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    fpaths = download(scrape=args.scrape, datasets=args.datasets, dest=args.out)
