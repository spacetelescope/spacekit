.. _radio:

************************
spacekit.extractor.radio
************************

.. currentmodule:: spacekit.extractor.radio


Querying and downloading .fits files from a MAST s3 bucket on AWS. Unlike `spacekit.extractor.scrape <spacekit.extractor.scrape>`_, which can access data in private s3 buckets, this module is specifically for collecting data from the publicly available MAST website and/or MAST data hosted on s3. Instead of scraping a closed collection, you're receiving data from an open channel - like a radio.

.. autoclass:: Radio
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HstSvmRadio
    :members:
