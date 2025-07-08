from ast import Assert
from pytest import mark
from spacekit.skopes.jwst.cal.predict import JwstCalPredict, predict_handler


EXPECTED = {
    "jw01018-o006_niriss": {'gbSize': 1.39},
    "jw02732-o001_nircam": {'gbSize': 3.54},
    "jw02732-o005_miri": {'gbSize': 2.58},
    "jw01022-o016_nirspec": {'gbSize': 3.16},
    "jw01192-o011_miri_ch1": {'gbSize': 2.98},
    "jw01192-o011_miri_ch3": {'gbSize': 2.89},
    "jw01309-o023_nircam": {'gbSize': 6.73},
}


@mark.jwst
@mark.predict
def test_jwst_cal_predict(jwstcal_input_path):
    jcal = JwstCalPredict(input_path=jwstcal_input_path)
    assert jcal.img3_reg.__name__ == "Builder"
    assert jcal.model_path == 'models/jwst_cal'
    assert jcal.tx_file == 'models/jwst_cal/tx_data-{}.json'
    assert jcal.img3_reg.blueprint == "jwst_img3_reg"
    assert jcal.img3_reg.model.name == 'img3_reg'
    assert len(jcal.img3_reg.model.layers) == 10
    assert jcal.spec3_reg.blueprint == "jwst_spec3_reg"
    assert jcal.spec3_reg.model.name == 'spec3_reg'
    assert len(jcal.spec3_reg.model.layers) == 11
    jcal.run_inference()
    assert jcal.input_data['IMAGE'].shape == (3, 27)
    assert jcal.inputs['IMAGE'].shape == (3, 18)
    assert jcal.input_data['SPEC'].shape == (8, 27)
    assert jcal.inputs['SPEC'].shape == (8, 18)
    for k, v in jcal.predictions.items():
        name = '_'.join([k.split('_')[0], k.split("_")[2]])
        sfx = k.split('_')[-1].split('-')[0]
        if sfx in ['ch1', 'ch3']:
            name += f'_{sfx}'
        assert round(EXPECTED[name]["gbSize"],1) == round(v['gbSize'],1)

@mark.jwst
@mark.predict
def test_jwst_cal_predict_handler(jwstcal_input_path):
    jcal = predict_handler(jwstcal_input_path)
    assert len(jcal.predictions) == 11
    for k, v in jcal.predictions.items():
        name = '_'.join([k.split('_')[0], k.split("_")[2]])
        sfx = k.split('_')[-1].split('-')[0]
        if sfx in ['ch1', 'ch3']:
            name += f'_{sfx}'
        assert round(EXPECTED[name]["gbSize"],1) == round(v['gbSize'],1)


@mark.jwst
@mark.predict
def test_jwst_cal_predict_input_single_file_path(jwstcal_input_path):
    filename = "jw02732005001_02105_00001_mirimage_uncal.fits"
    jwstcal_file_path = f"{jwstcal_input_path}/{filename}"
    jcal = JwstCalPredict(input_path=jwstcal_file_path)
    jcal.preprocess()
    assert jcal.input_path == 'tests/data/jwstcal/predict/inputs'
    assert jcal.pid == "jw02732005"
    assert len(jcal.input_data['IMAGE']) == 1
    assert jcal.input_data['IMAGE'].nexposur.values[0] == 2


@mark.jwst
@mark.predict
def test_jwst_cal_predict_pid_obs(jwstcal_input_path):
    """Test selecting specific observation number within a program with multiple observations"""
    # case 1: obs as integer value
    jcal1 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs=1)
    jcal1.preprocess()
    assert len(jcal1.input_data['IMAGE']) == 1
    assert jcal1.input_data['IMAGE'].index[0] == 'jw02732-o001_t1_nircam_clear-f150w'
    assert jcal1.input_data['IMAGE']['nexposur'].values[0] == 4
    del jcal1
    # case 2: obs as 3 character numeric string (leading zeros)
    jcal2 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs="005")
    # case 3: obs as single character numeric string
    jcal3 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs="5")
    # cases 2 and 3 should produce the same results:
    for jcal in [jcal2, jcal3]:
        jcal.preprocess()
        assert len(jcal.input_data['IMAGE']) == 1
        assert jcal.input_data['IMAGE'].index[0] == 'jw02732-o005_t1_miri_f1130w'
        assert jcal.input_data['IMAGE']['nexposur'].values[0] == 2
    del jcal2, jcal3
    # case 4: invalid obsnum triggers warning, resets obs to empty string and collects entire program ID
    jcal4 = JwstCalPredict(input_path=jwstcal_input_path, pid=2732, obs="bad")
    jcal4.preprocess()
    assert jcal4.obs == ''
    assert len(jcal4.input_data['IMAGE']) == 2


@mark.jwst
@mark.predict
def test_jwst_cal_predict_fnf_exception():
    try:
        jcal = JwstCalPredict(input_path="nonexistent/path")
        jcal.preprocess()
    except FileNotFoundError:
        assert True


@mark.jwst
@mark.predict
def test_jwst_cal_predict_radec_nans_skip(jwstcal_input_path):
    input_path = jwstcal_input_path + "/nans"
    jcal = JwstCalPredict(input_path=input_path)
    jcal.preprocess()
    # should ignore the bad exposure (NaN ra_ref val) but keep nrs2 exp
    assert jcal.inputs['SPEC'].shape == (1, 18)
    assert jcal.input_data['SPEC'].loc['jw01022-o016_t1_nirspec_g140h-f100lp']['nexposur'] == 1
    # nrs2 only (nrs1 exposure was removed)
    assert jcal.input_data['SPEC'].loc['jw01022-o016_t1_nirspec_g140h-f100lp']['detector'] == 16