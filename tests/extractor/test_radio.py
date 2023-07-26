from pytest import mark
from spacekit.extractor.radio import HstSvmRadio


@mark.hst
@mark.svm
@mark.extractor
@mark.radio
def test_scrape_mast(scraped_fits_data):
    scraper = HstSvmRadio(scraped_fits_data)
    scraper.scrape_mast()
    assert scraper.df.shape == (1, 15)
    assert "category" in scraper.df.columns
