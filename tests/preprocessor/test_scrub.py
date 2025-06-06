from pytest import mark
from spacekit.preprocessor.scrub import HstSvmScrubber, JwstCalScrubber
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA
import os
import pandas as pd
import numpy as np

SVM_SCRUBBED_COLS = [
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
]

SVM_FINAL_COLS = [
    "numexp",
    "rms_ra",
    "rms_dec",
    "nmatches",
    "point",
    "segment",
    "gaia",
    "det",
    "wcs",
    "cat",
]

JWST_EXPECTED = {
    'jw02732-o001-t2_nircam_clear-f150w': [
        'jw02732001005_02103_00005_nrcb1',
        'jw02732001005_02103_00005_nrcb2',
        'jw02732001005_02103_00005_nrcb3',
        'jw02732001005_02103_00005_nrcb4'
    ],
    "jw02732-o005-t1_miri_f1130w": [
        'jw02732005001_02105_00001_mirimage',
        'jw02732005001_02105_00002_mirimage'
    ],
    'jw01018-o006-t1_niriss_clear-f150w': [
        'jw01018006001_02101_00002_nis_uncal.fits',
        'jw01018006001_02101_00003_nis_uncal.fits',
        'jw01018006001_02101_00004_nis_uncal.fits',
        'jw02732001005_02103_00005_nrcb2_uncal.fits',
    ],
}

JWST_SCRUBBED_COLS = [
    'instr',
    'detector',
    'exp_type',
    'visitype',
    'filter',
    'pupil',
    'grating',
    'fxd_slit',
    'channel',
    'subarray',
    'bkgdtarg',
    'is_imprt',
    'tsovisit',
    'nexposur',
    'numdthpt',
    'band',
    'targ_max_offset',
    'offset',
    'max_offset',
    'mean_offset',
    'sigma_offset',
    'err_offset',
    'sigma1_mean',
    'frac',
    'targ_frac',
    'gs_mag',
    'crowdfld'
]


@mark.hst
@mark.svm
@mark.preprocessor
@mark.scrub
def test_svm_scrubber(raw_svm_data, single_visit_path):
    scrubber = HstSvmScrubber(
        single_visit_path,
        data=raw_svm_data,
        output_path="tmp",
        output_file="scrubbed",
        crpt=0,
    )
    assert scrubber.df.shape[1] == 9
    scrubber.preprocess_data()
    assert scrubber.df.shape[1] == 10
    assert list(scrubber.df.columns) == SVM_FINAL_COLS
    assert os.path.exists(scrubber.data_path)
    base_path = os.path.dirname(scrubber.data_path)
    raw_file = "raw_" + os.path.basename(scrubber.data_path)
    raw_fpath = os.path.join(base_path, raw_file)
    assert os.path.exists(raw_fpath)


# TEST SCRUBCOLS
@mark.hst
@mark.svm
@mark.preprocessor
@mark.scrub
def test_svm_scrub_cols(raw_svm_data, single_visit_path):
    scrubber = HstSvmScrubber(
        single_visit_path,
        data=raw_svm_data,
        output_path="tmp",
        output_file="scrubbed",
        crpt=0,
    )
    scrubber.scrub_columns()
    assert scrubber.df.shape == (1, 10)
    for col in SVM_SCRUBBED_COLS:
        if col in list(scrubber.df.columns):
            assert True
        else:
            assert False


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber(jwstcal_input_path):
    scrubber = JwstCalScrubber(jwstcal_input_path, encoding_pairs=KEYPAIR_DATA)
    assert len(scrubber.fpaths) == 26
    assert len(scrubber.imgpix) == 3
    assert len(scrubber.specpix) == 8
    for (vi, vs) in list(zip(scrubber.imgpix.values(), scrubber.specpix.values())):
        assert len(vi.keys()) == 48
        assert len(vs.keys()) == 48
    for exp in ["IMAGE", "SPEC"]:
        inputs = scrubber.scrub_inputs(exp_type=exp)
        assert len(inputs) == 3 if exp == "IMAGE" else 8
        assert list(inputs.columns) == JWST_SCRUBBED_COLS


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_imgname_fltr_only_or_none(jwst_cal_img_df):
    # pupil == NONE : optelem = f"{fltr}"
    np_exp = ['jw01023-o011_miri_f560w']
    no_pupil = jwst_cal_img_df.loc[
        (jwst_cal_img_df.PUPIL == "NONE") & (jwst_cal_img_df['SUBARRAY'].isin(["FULL", "NONE"]))
    ]
    optelems_filters = list(zip(
        [n.split('_')[-1] for n in list(no_pupil.index)], 
        [f.lower() for f in no_pupil['FILTER'].values]
    ))
    assert list(set([o == f for o, f in optelems_filters])) == [True]
    assert np_exp == sorted(list(no_pupil['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_imgname_pupil_first(jwst_cal_img_df):
    # pupil == clear/clearp/405n: optelem =  f"{pupil}-{fltr}"
    pf_exp = [
        'jw01088-o003_niriss_clearp-f356w', 
        'jw01245-o037_nircam_clear-f070w-sub400p',
        'jw02514-o265_nircam_clear-f150w',
        'jw02514-o265_nircam_clear-f356w',
        'jw04453-o001_nircam_f405n-f444w'
    ]
    pupil_first = jwst_cal_img_df.loc[jwst_cal_img_df['PUPIL'].isin(["CLEAR", "CLEARP", "F405N"])]
    optelems_pfs = list(zip(
        [n.split('_')[-1] for n in list(pupil_first.index)],
        [p.lower() for p in pupil_first['PUPIL'].values],
        [f.lower() for f in pupil_first['FILTER'].values]
    ))
    assert list(set([o.split('-')[0] == p for o, p, _ in optelems_pfs])) == [True]
    assert list(set([o.split('-')[1] == f for o, _, f in optelems_pfs])) == [True]
    assert pf_exp == sorted(list(pupil_first['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_imgname_fltr_first(jwst_cal_img_df):
    #  pupil != none/clear/clearp/405n: optelem = f"{fltr}-{pupil}"
    ff_exp = ['jw01086-o002_niriss_clear-f200w-wfss64r', 'jw04453-o002_nircam_f322w2-f323n']
    filter_first = jwst_cal_img_df.loc[~jwst_cal_img_df['PUPIL'].isin(["CLEAR", "CLEARP", "F405N", "NONE"])]
    optelems_ffs = list(zip(
        [n.split('_')[-1] for n in list(filter_first.index)],
        [f.lower() for f in filter_first['FILTER'].values],
        [p.lower() for p in filter_first['PUPIL'].values]
    ))
    assert list(set([o.split('-')[0] == f for o, f, _ in optelems_ffs])) == [True]
    assert list(set([o.split('-')[1] == p for o, _, p in optelems_ffs])) == [True]
    assert ff_exp == sorted(list(filter_first['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_imgname_subarrays(jwst_cal_img_df):
    sb_exp = ['jw01086-o002_niriss_clear-f200w-wfss64r', 'jw01245-o037_nircam_clear-f070w-sub400p']
    subarrays = jwst_cal_img_df.loc[~jwst_cal_img_df['SUBARRAY'].isin(["NONE", "FULL"])]
    optelems_sbs = list(zip(
        [n.split('_')[-1] for n in list(subarrays.index)],
        [s.lower() for s in subarrays['SUBARRAY'].values],
    ))
    assert list(set([o.split('-')[-1] == s for o, s in optelems_sbs])) == [True]
    assert sb_exp == sorted(list(subarrays['name'].values))

@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_wfsc_img(jwst_cal_wfsc_data):
    scrubber = JwstCalScrubber(
        "tmp",
        data=jwst_cal_wfsc_data, 
        mode='df', 
        encoding_pairs=KEYPAIR_DATA
    )
    # 'jw04509-o102_t1_nircam_f212n-wlm8-nrca1'
    df = pd.DataFrame.from_dict(scrubber.imgpix, orient='index')
    assert len(df) == 1
    name = list(df.index)[0]
    optelem = '-'.join([df.iloc[0]['FILTER'].lower(), df.iloc[0]['PUPIL'].lower()])
    assert optelem == '-'.join(name.split('_')[-1].split('-')[:-1])
    assert name.split('-')[-1] == df.iloc[0]['DETECTOR'].lower()
    assert df['NEXPOSUR'].iloc[0] == 2


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_specname_fltr_first(jwst_cal_spec_df):
    # optelem = f"{fltr}-{pupil}"
    fp = jwst_cal_spec_df.loc[
        (
            jwst_cal_spec_df['PUPIL'] != "NONE"
        ) & (
            jwst_cal_spec_df['EXP_TYPE'].isin(["NRC_WFSS", "NIS_SOSS", "NRC_TSGRISM"])
        )
    ]
    fp_exp = ['jw01091-o002_niriss_f277w-gr700xd-substrip256', 'jw01309-o023_nircam_f322w2-grismr']
    optelems_fp = list(zip(
        [n.split('_')[-1] for n in list(fp.index)],
        [f.lower() for f in fp['FILTER'].values],
        [p.lower() for p in fp['PUPIL'].values]
    ))
    assert list(set([o.split('-')[0] == f for o, f, _ in optelems_fp])) == [True]
    assert list(set([o.split('-')[1] == p for o, _, p in optelems_fp])) == [True]
    assert fp_exp == sorted(list(fp['name'].values))

    # optelem = f"{fltr}-{grating}"
    fg = jwst_cal_spec_df.loc[
        (
            jwst_cal_spec_df['GRATING'] != "NONE"
        ) & (
            jwst_cal_spec_df['EXP_TYPE'] != "NRS_IFU"
        )
    ]
    fg_exp = [
        'jw01117-o031_nirspec_clear-prism',
        'jw01118-o002_nirspec_f100lp-g140h-s200a2-allslits',
        'jw02677-o002_nirspec_f290lp-g395m',
        'jw04498-o121_nirspec_clear-prism-s1600a1-sub512'
    ]
    optelems_fg = list(zip(
        [n.split('_')[-1] for n in list(fg.index)],
        [f.lower() for f in fg['FILTER'].values],
        [g.lower() for g in fg['GRATING'].values]
    ))
    assert list(set([o.split('-')[0] == f for o, f, _ in optelems_fg])) == [True]
    assert list(set([o.split('-')[1] == g for o, _, g in optelems_fg])) == [True]
    assert fg_exp == sorted(list(fg['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_specname_pupil_first(jwst_cal_spec_df):
    # optelem = f"{pupil}-{fltr}"
    pf = jwst_cal_spec_df.loc[
        (
            jwst_cal_spec_df['PUPIL'] != "NONE"
        ) & (
            ~jwst_cal_spec_df['EXP_TYPE'].isin(["NRC_WFSS", "NIS_SOSS", "NRC_TSGRISM"])
        )
    ]
    pf_exp = ['jw02079-o004_niriss_f200w-gr150r', 'jw02738-o004_niriss_f200w-gr150c']

    optelems_pf = list(zip(
        [n.split('_')[-1] for n in list(pf.index)],
        [p.lower() for p in pf['PUPIL'].values],
        [f.lower() for f in pf['FILTER'].values]
    ))
    assert list(set([o.split('-')[0] == p for o, p, _ in optelems_pf])) == [True]
    assert list(set([o.split('-')[1] == f for o, _, f in optelems_pf])) == [True]
    assert pf_exp == sorted(list(pf['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_specname_grating_first(jwst_cal_spec_df):
    # optelem = f"{grating}-{fltr}"
    gf = jwst_cal_spec_df.loc[
        (
            jwst_cal_spec_df['GRATING'] != "NONE"
        ) & (
            jwst_cal_spec_df['EXP_TYPE'] == "NRS_IFU"
        )
    ]
    gf_exp = ['jw01118-o006_nirspec_g140m-f100lp']
    optelems_gf = list(zip(
        [n.split('_')[-1] for n in list(gf.index)],
        [g.lower() for g in gf['GRATING'].values],
        [f.lower() for f in gf['FILTER'].values]
    ))
    assert list(set([o.split('-')[0] == g for o, g, _ in optelems_gf])) == [True]
    assert list(set([o.split('-')[1] == f for o, _, f in optelems_gf])) == [True]
    assert gf_exp == sorted(list(gf['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_specname_fltr_only_or_none(jwst_cal_spec_df):
    no_opt = jwst_cal_spec_df.loc[jwst_cal_spec_df['FILTER'] == "NONE"]
    no_exp = ['jw01023-o013_miri_ch1-short', 'jw01023-o013_miri_ch3-short'] # mir_mrs
    assert list(set([n.split('_')[-1] in ["ch1-short", "ch3-short"] for n in list(no_opt.index)])) == [True]
    assert no_exp == sorted(list(no_opt['name'].values))

    fo_exp = ['jw01029-o007_miri_p750l']
    fo = jwst_cal_spec_df.loc[
        (
            jwst_cal_spec_df['FILTER'] != "NONE"
        ) & (
            jwst_cal_spec_df['GRATING'] == "NONE"
        ) & (
            jwst_cal_spec_df['PUPIL'] == "NONE"
        )
    ]
    optelems_fo = list(zip(
        [n.split('_')[-1] for n in list(fo.index)],
        [f.lower() for f in fo['FILTER'].values]
    ))
    assert list(set([o.split('-')[0] == f for o, f in optelems_fo])) == [True]
    assert fo_exp == sorted(list(fo['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_specname_slit_subarr(jwst_cal_spec_df):
    # subarray != "FULL", "NONE": subarray = optelem[-1]
    sb_exp = ['jw01091-o002_niriss_f277w-gr700xd-substrip256']
    sb = jwst_cal_spec_df.loc[
        (
            ~jwst_cal_spec_df['SUBARRAY'].isin(["NONE", "FULL"])
        ) & (
            jwst_cal_spec_df['FXD_SLIT'] == "NONE"
        )
    ]
    optelems_sbs = list(zip(
        [n.split('_')[-1] for n in list(sb.index)],
        [s.lower() for s in sb['SUBARRAY'].values],
    ))
    assert list(set([o.split('-')[-1] == s for o, s in optelems_sbs])) == [True]
    assert sb_exp == sorted(list(sb['name'].values))

    ss_exp = ['jw01118-o002_nirspec_f100lp-g140h-s200a2-allslits', 'jw04498-o121_nirspec_clear-prism-s1600a1-sub512']
    ss = jwst_cal_spec_df.loc[
        (
            ~jwst_cal_spec_df['SUBARRAY'].isin(["NONE", "FULL"])
        ) & (
            jwst_cal_spec_df['FXD_SLIT'] != "NONE"
        )
    ]
    slit_subs = list(zip(
        [n.split('_')[-1] for n in list(ss.index)],
        [slit.lower() for slit in ss['FXD_SLIT'].values],
        [sub.lower() for sub in ss['SUBARRAY'].values],
    ))
    assert list(set([o.split('-')[-2] == slit for o, slit, _ in slit_subs])) == [True]
    assert list(set([o.split('-')[-1] == sub for o, _, sub in slit_subs])) == [True]
    assert ss_exp == sorted(list(ss['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber_l3_tac_name(jwst_cal_tac_df):
    tso_ami_coron = ["MIR_4QPM", "MIR_LYOT", "NRC_CORON", "NIS_AMI", "NRS_BRIGHTOBJ", "NRC_TSGRISM", "NRC_TSIMAGE"]
    for name in list(jwst_cal_tac_df.index):
        exptype = jwst_cal_tac_df.loc[name]['EXP_TYPE']
        tsovisit = jwst_cal_tac_df.loc[name]['TSOVISIT']
        if not tsovisit:
            assert exptype in tso_ami_coron

    # only if tso
    miri_slitless = jwst_cal_tac_df.loc[jwst_cal_tac_df["EXP_TYPE"] == "MIR_LRS-SLITLESS"]
    assert list(set(miri_slitless['TSOVISIT'].unique())) == [True]

    # NRC_CORON append 'image3
    nrc_coron = jwst_cal_tac_df.loc[jwst_cal_tac_df['EXP_TYPE'] == "NRC_CORON"]
    assert nrc_coron.iloc[0]['name'].split('-')[-1] == 'image3'

    fo = jwst_cal_tac_df.loc[
        (jwst_cal_tac_df['GRATING'] == "NONE") & (jwst_cal_tac_df['PUPIL'] == "NONE")
    ]
    fo_exp = ['jw01029-o006_miri_p750l-slitlessprism', 'jw03730-o010_miri_f1500w-sub256']
    optelems_fo = list(zip(
        [n.split('_')[-1] for n in list(fo.index)],
        [f.lower() for f in fo['FILTER'].values]
    ))
    assert list(set([o.split('-')[0] == f for o, f in optelems_fo])) == [True]
    assert fo_exp == sorted(list(fo['name'].values))

    # optelem = f"{fltr}-{pupil}" NIS_SOSS, NRC_TSGRISM, NRC_TSIMAGE
    # NRC_TSIMAGE only TAC exp_type that follows IMAGE rules
    fp_exp =[
        'jw02183-o002_nircam_f212n-wlp8-subgrism256',
        'jw02183-o015_nircam_f444w-grismr-subgrism256',
        'jw04498-o024_niriss_clear-gr700xd-substrip256'
    ]
    fp = jwst_cal_tac_df.loc[
        (
            jwst_cal_tac_df['PUPIL'] != "NONE"
        ) & (
            jwst_cal_tac_df['EXP_TYPE'].isin(["NRC_WFSS", "NIS_SOSS", "NRC_TSGRISM", "NRC_TSIMAGE"])
        )
    ]
    optelems_fp = list(zip(
        [n.split('_')[-1] for n in list(fp.index)],
        [f.lower() for f in fp['FILTER'].values],
        [p.lower() for p in fp['PUPIL'].values]
    ))
    assert list(set([o.split('-')[0] == f for o, f, _ in optelems_fp])) == [True]
    assert list(set([o.split('-')[1] == p for o, _, p in optelems_fp])) == [True]
    assert fp_exp == sorted(list(fp['name'].values))

    pf = jwst_cal_tac_df.loc[
        (
            jwst_cal_tac_df['PUPIL'] != "NONE"
        ) & (
            ~jwst_cal_tac_df['EXP_TYPE'].isin(["NRC_WFSS", "NIS_SOSS", "NRC_TSGRISM", "NRC_TSIMAGE"])
        )
    ]
    pf_exp = ['jw01193-o037_nircam_maskrnd-f210m-image3']
    optelems_pf = list(zip(
        [n.split('_')[-1] for n in list(pf.index)],
        [p.lower() for p in pf['PUPIL'].values],
        [f.lower() for f in pf['FILTER'].values]
    ))
    assert list(set([o.split('-')[0] == p for o, p, _ in optelems_pf])) == [True]
    assert list(set([o.split('-')[1] == f for o, _, f in optelems_pf])) == [True]
    assert pf_exp == sorted(list(pf['name'].values))

    # NRS_BRIGHTOBJ
    slits = jwst_cal_tac_df.loc[jwst_cal_tac_df['FXD_SLIT'] != "NONE"]
    drop_slit = slits.loc[(slits['FILTER'] == "CLEAR") & (slits['GRATING'] == 'PRISM')]
    keep_slit = slits.loc[~slits.name.isin(drop_slit['name'])]
    assert drop_slit.iloc[0]['FXD_SLIT'].lower() not in drop_slit.iloc[0]['name']
    assert keep_slit.iloc[0]['name'].split('-')[-2] == keep_slit.iloc[0]['FXD_SLIT'].lower()

    sb = jwst_cal_tac_df.loc[
        (
            ~jwst_cal_tac_df['SUBARRAY'].isin(["NONE", "FULL"])
        ) & (
            jwst_cal_tac_df['FXD_SLIT'] == "NONE"
        )
    ]
    sb_exp = [
        'jw01029-o006_miri_p750l-slitlessprism',
        'jw02183-o002_nircam_f212n-wlp8-subgrism256',
        'jw02183-o015_nircam_f444w-grismr-subgrism256',
        'jw03730-o010_miri_f1500w-sub256',
        'jw04498-o024_niriss_clear-gr700xd-substrip256'
    ]
    optelems_sbs = list(zip(
        [n.split('_')[-1] for n in list(sb.index)],
        [s.lower() for s in sb['SUBARRAY'].values],
    ))
    assert list(set([o.split('-')[-1] == s for o, s in optelems_sbs])) == [True]
    assert sb_exp == sorted(list(sb['name'].values))


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_target_id_grouping(jwst_cal_img_data, jwst_cal_spec_data, jwst_cal_tac_data, jwst_cal_wfsc_data):
    data = pd.concat([jwst_cal_img_data, jwst_cal_wfsc_data, jwst_cal_spec_data, jwst_cal_tac_data], axis=0)
    scrubber = JwstCalScrubber(
        "tmp",
        data=data,
        mode='df',
        encoding_pairs=KEYPAIR_DATA,
    )
    df = pd.concat(
        [
            pd.DataFrame.from_dict(scrubber.imgpix, orient='index'),
            pd.DataFrame.from_dict(scrubber.specpix, orient='index'),
            pd.DataFrame.from_dict(scrubber.tacpix, orient='index')
        ], axis=0
    )
    assert len(df) == 29
    # source-based
    src_based = df.loc[df['EXP_TYPE'].isin(scrubber.source_based)]
    source_names = list(src_based.index)
    assert len(source_names) == 7
    assert list(set([s.split('_')[1] == 's000000001' for s in source_names])) == [True]
    df.drop(source_names, axis=0, inplace=True)
    df['dname'] = df.index
    df['tid'] = df['dname'].apply(lambda x: x.split('_')[1])
    tids = list(df['tid'].unique())

    tn, rn, gn = scrubber.fake_target_ids()
    for targname, tid in tn.items():
        assert list(df.loc[df['TARGNAME'] == targname]['tid'].unique()) == [tid]

    for targra, tid in rn.items():
        if tid in tids:
            assert list(df.loc[np.round(df['TARG_RA'], 6) == targra]['tid'].unique()) == [tid]

    for gsmag, tid in gn.items():
        assert list(df.loc[df['GS_MAG'] == gsmag]['tid'].unique()) == [tid]

    # PARALLEL_PURE
    assert list(df.loc[df['VISITYPE'] == 'PARALLEL_PURE']['tid'].unique()) == ['t0']


def test_jwst_cal_scrub_miri_ifu_names(jwst_cal_spec_data):
    data = jwst_cal_spec_data.loc[jwst_cal_spec_data['EXP_TYPE'] == "MIR_MRS"]
    data.loc['jw01023013001_02101_00004_mirifulong', 'BAND'] = 'MEDIUM'
    data.loc['jw01023013001_02101_00002_mirifushort', 'BAND'] = 'LONG'
    data.loc['jw01023013001_02101_00003_mirifulong', 'BAND'] = 'LONG'
    data.loc['jw01023013001_02101_00004_mirifushort', 'BAND'] = 'MEDIUM'
    scrubber = JwstCalScrubber(
        "tmp",
        data=data,
        mode='df',
        encoding_pairs=KEYPAIR_DATA,
    )
    names = sorted(list(scrubber.specpix.keys()))
    assert [n.split('_')[-1] for n in names] == ['ch1-long', 'ch1-medium', 'ch1-short', 'ch3-long', 'ch3-medium', 'ch3-short']
