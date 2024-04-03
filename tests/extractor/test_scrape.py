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

JWST_EXPECTED_HEADERS = dict(
    jw02732001005_02103_00005_nrcb1 = {
        'PROGRAM': '02732',
        'OBSERVTN': '001',
        'NEXPOSUR': 4,
        'BKGDTARG': 'F',
        'IS_IMPRT': 'NaN',
        'VISITYPE': 'PRIME_TARGETED_FIXED',
        'TSOVISIT': 'F',
        'TARGNAME': 'NGC 7320',
        'TARG_RA': 338.9983122537918,
        'TARG_DEC': 33.95798043577638,
        'INSTRUME': 'NIRCAM',
        'DETECTOR': 'NRCB1',
        'FILTER': 'F150W',
        'PUPIL': 'CLEAR',
        'GRATING': 'N/A',
        'FXD_SLIT': 'NaN',
        'EXP_TYPE': 'NRC_IMAGE',
        'CHANNEL': 'SHORT',
        'BAND': 'NaN',
        'SUBARRAY': 'FULL',
        'NUMDTHPT': 5,
        'GS_RA': 339.0885699921705,
        'GS_DEC': 34.03843427549464,
        'GS_MAG': 'NaN',
        'CROWDFLD': 'NaN',
        'RA_REF': 339.0433138619121,
        'DEC_REF': 33.93338350206477,
        'CRVAL1': 339.04331382287,
        'CRVAL2': 33.933384524872
    },
    jw02732005001_02105_00001_mirimage = {
        'PROGRAM': '02732',
        'OBSERVTN': '005',
        'NEXPOSUR': 2,
        'BKGDTARG': 'T',
        'IS_IMPRT': 'NaN',
        'VISITYPE': 'PRIME_TARGETED_FIXED',
        'TSOVISIT': 'F',
        'TARGNAME': 'NGC 7319 BG',
        'TARG_RA': 339.0321483333333,
        'TARG_DEC': 33.99454722222222,
        'INSTRUME': 'MIRI',
        'DETECTOR': 'MIRIMAGE',
        'FILTER': 'F1130W',
        'PUPIL': 'N/A',
        'GRATING': 'N/A',
        'EXP_TYPE': 'MIR_IMAGE',
        'FXD_SLIT': 'NaN',
        'CHANNEL': 'N/A',
        'BAND': 'NaN',
        'SUBARRAY': 'FULL',
        'NUMDTHPT': 2,
        'GS_RA': 339.061734363325,
        'GS_DEC': 34.16297426913081,
        'GS_MAG': 'NaN',
        'CROWDFLD': 'NaN',
        'RA_REF': 339.0392914656005,
        'DEC_REF': 34.01419185596434,
        'CRVAL1': 339.03931377347,
        'CRVAL2': 34.014195549057
    },
    jw01018006001_02101_00001_nis = {
        'PROGRAM': '01018',
        'OBSERVTN': '006',
        'NEXPOSUR': 4,
        'BKGDTARG': False,
        'IS_IMPRT': 'NaN',
        'VISITYPE': 'PRIME_TARGETED_FIXED',
        'TSOVISIT': False,
        'TARGNAME': 'Gaia DR1 4657988102378365184',
        'TARG_RA': 80.48876249999999,
        'TARG_DEC': -69.49941666666666,
        'INSTRUME': 'NIRISS',
        'DETECTOR': 'NIS',
        'FILTER': 'CLEAR',
        'PUPIL': 'F150W',
        'GRATING': 'N/A',
        'FXD_SLIT': 'NaN',
        'EXP_TYPE': 'NIS_IMAGE',
        'CHANNEL': 'NONE',
        'BAND': 'NaN',
        'SUBARRAY': 'FULL',
        'NUMDTHPT': 4,
        'GS_RA': 80.3457270042323,
        'GS_DEC': -69.52326084117196,
        'GS_MAG': 'NaN',
        'CROWDFLD': 'NaN',
        'RA_REF': 80.65048990317743,
        'DEC_REF': -69.46002290849921,
        'CRVAL1': 80.650493681119,
        'CRVAL2': -69.459995044253
    }
)

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


@mark.jwst
@mark.extractor
@mark.scrape
def test_jwst_cal_scraper(jwstcal_input_path):
    scraper = JwstFitsScraper(jwstcal_input_path, data=None, sfx="_uncal.fits")
    assert len(scraper.fpaths) == 10
    exp_headers = scraper.scrape_fits()
    assert len(exp_headers) == 10
    assert sorted(list(exp_headers.keys())) == [
        'jw01018006001_02101_00001_nis',
        'jw01018006001_02101_00002_nis',
        'jw01018006001_02101_00003_nis',
        'jw01018006001_02101_00004_nis',
        'jw02732001005_02103_00005_nrcb1',
        'jw02732001005_02103_00005_nrcb2',
        'jw02732001005_02103_00005_nrcb3',
        'jw02732001005_02103_00005_nrcb4',
        'jw02732005001_02105_00001_mirimage',
        'jw02732005001_02105_00002_mirimage'
    ]
    keys = [
        'jw02732005001_02105_00001_mirimage',
        'jw02732001005_02103_00005_nrcb1',
        'jw01018006001_02101_00001_nis'
    ]
    for key in keys:
        for k, v in exp_headers[key].items():
            assert JWST_EXPECTED_HEADERS[key][k] == v
