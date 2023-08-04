from pytest import mark
from spacekit.extractor.scrape import JsonScraper, SvmFitsScraper, JwstFitsScraper
import os

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

JWST_EXPECTED_HEADERS = []


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


@mark.hst
@mark.svm
@mark.extractor
@mark.scrape
def test_scrape_drizzle_fits(scrubbed_svm_data, single_visit_path):
    scraper = SvmFitsScraper(scrubbed_svm_data, single_visit_path)
    assert scraper.fpaths == {
        "hst_12286_38_wfc3_ir_total_ibl738": f"{single_visit_path}/ibl738/hst_12286_38_wfc3_ir_total_ibl738_drz.fits"
    }
    scraper.scrape_drizzle_fits()
    assert scraper.df.shape == (1, 14)
    for col in FITS_COL_EXPECTED:
        if col in list(scraper.df.columns):
            assert True
        else:
            assert False


#TODO
@mark.jwst
@mark.extractor
@mark.scrape
def test_jwst_cal_scraper(skope, input_path):
    scraper = JwstFitsScraper(input_path, data=None, sfx="_uncal.fits")
    assert len(scraper.fpaths) > 0
    exp_headers = scraper.scrape_fits()
