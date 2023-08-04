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


@mark.jwst
@mark.predict
def test_jwst_cal_predict(jwstcal_input_path):
    jcal = JwstCalPredict(input_path=jwstcal_input_path)
    assert jcal.img3_reg.__name__ == "Builder"
    assert jcal.img3_reg.blueprint == "jwst_img3_reg"
    assert jcal.img3_reg.model_path == 'models/jwst_cal/img3_reg'
    assert jcal.tx_file == 'models/jwst_cal/tx_data.json'
    assert jcal.img3_reg.model.name == 'img3_reg'
    jcal.run_inference()


@mark.jwst
@mark.predict
def test_jwst_cal_predict_handler(jwstcal_input_path):
    preds = predict_handler(jwstcal_input_path)
