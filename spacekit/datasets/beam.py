"""
Retrieve dataset archive (.zip) files from the web, an s3 bucket, or local disk.
This script is primarily intended as a retrieval and extraction step before launching spacekit.dashboard
For more customized control of dataset retrieval (such as your own), use the spacekit.extractor.scrape module.

Examples:
"datasets": if set, chooses specific archive dataset filenames to retrieve and extract

src: "web" -> Fetch and extract from one of the spacekit data archives:
archive: name of collection (see ``spacekit.datasets.meta``)
download(scrape="web:calcloud")
download(scrape="web:svm")

src: "file" -> Fetch and extract from path on local disk
archive: path
download(scrape="file:another/path/to/data)

src: "s3" -> Fetch and extract from an S3 bucket on AWS
archive: bucketname
"datasets" -> string of S3 archive file basenames separated by comma (without the .zip or .tgz suffix)
download(scrape="s3:mybucketname", "2021-11-04-1636048291,2021-10-28-1635457222,2021-08-22-1629663047")

archive: json filepath containing metadata structured similar to dictionaries in ``spacekit.datasets.meta``
"""
import argparse
import sys
import os
import json
from spacekit.datasets.meta import spacekit_collections
from spacekit.extractor.scrape import WebScraper, S3Scraper, FileScraper
from spacekit.logger.log import SPACEKIT_LOG

S3PREFIX = os.environ.get("PFX", "archive")


def download(scrape="file:data", datasets="2022-02-14,2021-11-04,2021-10-28", dest="."):
    if len(dest.split("/")) > 1:
        cache_dir = os.path.abspath(dest.split("/")[0])
        cache_subdir = "/".join(dest.split("/")[1:])
    elif dest in [None, "none", "None"]:
        cache_dir, cache_subdir ="~", "data"
    elif dest in [".", "data"]:
        cache_dir, cache_subdir = ".", "data"
    else:
        cache_dir, cache_subdir = dest, "data"
    src, archive = scrape.split(":")
    datasets = datasets.split(",")
    if src == "web":
        if archive.split(".")[-1] == "json":
            SPACEKIT_LOG.info("Scraping web via custom json file")
            with open(archive, "r") as j:
                collection = json.load(j)
                scraper = WebScraper(
                    collection["uri"],
                    collection["data"],
                    cache_dir=cache_dir,
                    cache_subdir=cache_subdir,
                )
        elif archive in spacekit_collections.keys():
            SPACEKIT_LOG.info(f"Scraping spacekit collection {archive.upper()}")
            cc = spacekit_collections[archive]  # "calcloud", "svm"
            dd = {}
            for d in datasets:
                dd[d] = cc["data"][d]
            scraper = WebScraper(
                cc["uri"], dd, cache_dir=cache_dir, cache_subdir=cache_subdir
            )
        else:
            SPACEKIT_LOG.error(
                f"Must use custom json file or one of the spacekit collections: {list(spacekit_collections.keys())}"
            )
    elif src == "s3":
        SPACEKIT_LOG.info("Scraping S3")
        scraper = S3Scraper(
            archive, pfx=S3PREFIX, cache_dir=cache_dir, cache_subdir=cache_subdir
        )
        scraper.make_s3_keys(fnames=datasets)
    elif src == "file":
        SPACEKIT_LOG.info("Scraping local directory")
        p = [f"{archive}/*.zip", f"{archive}/*"]
        scraper = FileScraper(
            patterns=p, clean=False, cache_dir=cache_dir, cache_subdir=cache_subdir
        )
    else:
        SPACEKIT_LOG.error("Unrecognized scrape arg. Must begin with web, s3, or file.")
    if scraper:
        try:
            scraper.scrape()
            if scraper.fpaths:
                SPACEKIT_LOG.info(f"Datasets: {scraper.fpaths}")
                return scraper.fpaths
        except Exception as e:
            SPACEKIT_LOG.error(f"Could not locate datasets {e}")
            sys.exit(1)
    elif fpaths:
        return fpaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scrape",
        default="web:calcloud",
        help="Uses a key:uri format where options for the key are limited to web, s3, or file. \
        The uri could be your own custom location if not using the default datasets.  \
        Examples are web:calcloud, web:custom.json, s3:mybucket, file:myfolder. \
        Visit spacekit.readthedocs.io for more info.",
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
