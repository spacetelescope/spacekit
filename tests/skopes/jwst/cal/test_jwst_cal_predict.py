from pytest import mark
from spacekit.skopes.jwst.cal.predict import JwstCalPredict, predict_handler


EXPECTED = {
    "niriss": {"gbSize": 2.57},
    "miri": {"gbSize": 0.79},
    "nircam": {"gbSize": 3.8},
}


@mark.jwst
@mark.predict
def test_jwst_cal_predict(jwstcal_input_path):
    jcal = JwstCalPredict(input_path=jwstcal_input_path)
    assert jcal.img3_reg.__name__ == "Builder"
    assert jcal.img3_reg.blueprint == "jwst_img3_reg"
    assert jcal.img3_reg.model_path == 'models/jwst_cal/img3_reg/img3_reg.keras'
    assert jcal.tx_file == 'models/jwst_cal/img3_reg/tx_data.json'
    assert jcal.img3_reg.model.name == 'img3_reg'
    assert len(jcal.img3_reg.model.layers) == 10
    jcal.run_inference()
    assert len(jcal.input_data['IMAGE']) == 3
    assert jcal.inputs['IMAGE'].shape == (3, 18)
    for k, v in jcal.predictions.items():
        instr = k.split("_")[1]
        assert EXPECTED[instr]["gbSize"] == v["gbSize"]


@mark.jwst
@mark.predict
def test_jwst_cal_predict_handler(jwstcal_input_path):
    jcal = predict_handler(jwstcal_input_path)
    assert len(jcal.predictions) == 3
    for k, v in jcal.predictions.items():
        instr = k.split("_")[1]
        assert EXPECTED[instr]["gbSize"] == v["gbSize"]
