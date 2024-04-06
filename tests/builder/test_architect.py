from pytest import mark
from conftest import check_skope
from spacekit.builder.architect import BuilderEnsemble, Builder
from spacekit.skopes.hst.svm.train import load_ensemble_data


@mark.hst
@mark.svm
@mark.builder
@mark.architect
def test_svm_ensemble_builder_with_data(skope, labeled_dataset, svm_train_npz):
    check_skope(skope, "svm")
    _, XTR, YTR, XTS, YTS, _, _ = load_ensemble_data(
        labeled_dataset,
        svm_train_npz,
        img_size=128,
        norm=0,
        v=0.85,
        output_path="tmp",
    )
    params = dict(
        batch_size=14,
        epochs=1,
        lr=1e-4,
        decay=[100000, 0.96],
        early_stopping=None,
        verbose=2,
        ensemble=True,
    )
    ens = BuilderEnsemble(
        X_train=XTR,
        y_train=YTR,
        X_test=XTS,
        y_test=YTS,
        params=params,
        input_name="svm_mixed_inputs",
        output_name="svm_output",
        name="ens_build_test",
    )
    assert ens.steps_per_epoch > 0
    ens.build()
    assert ens.model is not None


@mark.hst
@mark.svm
@mark.builder
@mark.architect
def test_svm_ensemble_builder_without_data(skope):
    check_skope(skope, "svm")
    ens = BuilderEnsemble()
    assert ens.blueprint == "ensemble"
    assert ens.steps_per_epoch > 0
    ens.load_saved_model(arch="svm_align", keras_archive=True)
    assert ens.model is not None
    assert len(ens.model.layers) == 28
    assert ens.model_path == "models/svm_align/ensembleSVM.keras"
    ens.find_tx_file()
    assert ens.tx_file == "models/svm_align/tx_data.json"


@mark.hst
@mark.cal
@mark.builder
@mark.architect
def test_hst_cal_builder_without_data(skope):
    check_skope(skope, "hstcal")
    builder = Builder(name="mem_clf")
    builder.load_saved_model(arch="hst_cal", keras_archive=True)
    assert builder.model is not None
    assert builder.blueprint == "hst_mem_clf"
    builder.get_blueprint(builder.blueprint)
    assert len(builder.model.layers) == 8
    assert builder.model_path == "models/hst_cal/mem_clf.keras"
    builder.find_tx_file()
    assert builder.tx_file == "models/hst_cal/tx_data.json"


@mark.jwst
@mark.builder
@mark.architect
def test_jwst_cal_builder_without_data(skope):
    check_skope(skope, "jwstcal")
    builder = Builder(name="img3_reg")
    builder.load_saved_model("jwst_cal")
    assert builder.model is not None
    assert builder.blueprint == "jwst_img3_reg"
    builder.get_blueprint(builder.blueprint)
    assert len(builder.model.layers) == 8
    assert builder.model_path == "models/jwst_cal/img3_reg"
    builder.find_tx_file()
    assert builder.tx_file == "models/jwst_cal/tx_data.json"
