from pytest import mark
from spacekit.extractor.scrape import JsonScraper, FitsScraper, MastScraper
import os
import pandas as pd

JSON_COL_EXPECTED = [
    "header.TARGNAME",
    "header.RA_TARG",
    "header.DEC_TARG",
    "header.NUMEXP",
    "gen_info.imgname",
    "Number_of_GAIA_sources.Number_of_GAIA_sources",
    "number_of_sources.detector",
    "number_of_sources.point",
    "number_of_sources.segment",
]

FITS_COL_EXPECTED = [
    "detector",
    "dataset",
    "targname",
    "ra_targ",
    "dec_targ",
    "numexp",
    "imgname",
    "point",
    "segment",
    "gaia",
    "rms_ra",
    "rms_dec",
    "nmatches",
    "wcstype",
]


# TEST JSON SCRAPER
@mark.extractor
@mark.scrape
def test_load_from_h5(h5_data):
    data = JsonScraper(file_basename=h5_data).load_h5_file()
    assert data.shape == (1, 9)


@mark.extractor
@mark.scrape
def test_json_scraper(raw_csv_file, single_visit_path):
    output_path = "tmp"
    os.makedirs(output_path, exist_ok=True)
    fname = os.path.basename(raw_csv_file).split(".")[0]
    jsc = JsonScraper(
        search_path=single_visit_path,
        file_basename=fname,
        crpt=0,
        output_path=output_path,
    )
    jsc.json_harvester()
    assert jsc.data.shape == (1, 9)

    for col in JSON_COL_EXPECTED:
        if col in list(jsc.data.columns):
            assert True
        else:
            assert False

    jsc.h5store()
    assert os.path.exists(jsc.h5_file)


@mark.extractor
@mark.scrape
def test_scrape_fits(scrubbed_cols_file, single_visit_path):
    data = pd.read_csv(scrubbed_cols_file, index_col="index")
    scraper = FitsScraper(data, single_visit_path)
    assert scraper.drz_paths == {
        "hst_12286_38_wfc3_ir_total_ibl738": f"{single_visit_path}/ibl738/hst_12286_38_wfc3_ir_total_ibl738_drz.fits"
    }
    scraper.scrape_fits()
    assert scraper.df.shape == (1, 14)
    for col in FITS_COL_EXPECTED:
        if col in list(scraper.df.columns):
            assert True
        else:
            assert False


@mark.extractor
@mark.scrape
def test_scrape_mast(scraped_fits_file):
    data = pd.read_csv(scraped_fits_file, index_col="index")
    scraper = MastScraper(data)
    scraper.scrape_mast()
    assert scraper.df.shape == (1, 15)
    assert "category" in scraper.df.columns
