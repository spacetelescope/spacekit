from spacekit.datasets.hst_cal import calcloud_data, calcloud_uri
from spacekit.extractor.scrape import WebScraper

fpaths = WebScraper(calcloud_uri, calcloud_data).scrape_repo()