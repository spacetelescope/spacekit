from pytest import mark
from spacekit.skopes.jwst.cal.predict import JwstCalPredict, predict_handler


EXPECTED = {
    "products": [
        # 'jw01018-o006-t1_niriss_clear-f150w',
        'jw02732-o005-t1_miri_f1130w',
        'jw02732-o001-t2_nircam_clear-f150w',
    ],
    "exposures": {
        "nircam": [
            'jw02732001005_02103_00005_nrcb1',
            'jw02732001005_02103_00005_nrcb2',
            'jw02732001005_02103_00005_nrcb3',
            'jw02732001005_02103_00005_nrcb4'
        ],
        "miri": [
            'jw02732005001_02105_00001_mirimage',
            'jw02732005001_02105_00002_mirimage'
        ]
    },
    "predictions": {
        "jw02732-o005-t1_miri_f1130w'": {"gbSize": 2},
        "jw02732-o001-t2_nircam_clear-f150w": {"gbSize": 60},
    },
}

@mark.skip(reason="tbd for model")
@mark.jwst
@mark.predict
def test_jwst_cal_preprocess(jwstcal_input_path):
    kwargs = dict(
        model_path=None,
        models={},
        tx_file=None,
        norm=0,
        norm_cols=[]
    )
    predictor = JwstCalPredict(input_path=jwstcal_input_path, **kwargs)
    predictor.preprocess()
    for prod in list(predictor.products.keys()):
        assert prod in EXPECTED['products']
    nrc_product = 'jw02732-o001-t2_nircam_clear-f150w'
    miri_product = 'jw02732-o005-t1_miri_f1130w'
    nrc_exposures = sorted(list(predictor.products[nrc_product].keys()))
    assert nrc_exposures == EXPECTED['exposures']['nircam']
    miri_exposures = sorted(list(predictor.products[miri_product].keys()))
    assert miri_exposures == EXPECTED['exposures']['miri']


@mark.skip(reason="tbd for model")
@mark.jwst
@mark.predict
def test_jwst_cal_predict(jwstcal_input_path):
    kwargs = dict(
        model_path=None,
        models={},
        tx_file=None,
        norm=0,
        norm_cols=[]
    )
    preds = predict_handler(jwstcal_input_path, **kwargs)
