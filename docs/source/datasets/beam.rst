.. _beam:

**********************
spacekit.datasets.beam
**********************

.. currentmodule:: spacekit.datasets.beam

.. autofunction:: download

Retrieve dataset archive (.zip) files from the web, an s3 bucket, or local disk.
This script is primarily intended as a retrieval and extraction step before launching spacekit.dashboard
For more customized control of dataset retrieval (such as your own), use the spacekit.extractor.scrape module.

Examples:
"datasets": if set, chooses specific archive dataset filenames to retrieve and extract

src: "git" - Fetch and extract from one of the spacekit data archives:
archive: name of collection (see ``spacekit.datasets.meta``)
download(scrape="git:calcloud")
download(scrape="git:svm")

src: "file" - Fetch and extract from path on local disk
archive: path
download(scrape="file:another/path/to/data)

src: "s3" - Fetch and extract from an S3 bucket on AWS
archive: bucketname
"datasets" - string of S3 archive file basenames separated by comma (without the .zip or .tgz suffix)
download(scrape="s3:mybucketname", "2021-11-04-1636048291,2021-10-28-1635457222,2021-08-22-1629663047")