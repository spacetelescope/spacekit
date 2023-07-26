from pytest import mark
from conftest import check_skope
from spacekit.builder.architect import BuilderEnsemble, Builder
from spacekit.skopes.hst.svm.train import load_ensemble_data


@mark.svm
@mark.builder
@mark.architect
def test_ensemble_builder_with_data(skope, labeled_dataset, svm_train_npz):
    check_skope(skope, "svm")
    tv_idx, XTR, YTR, XTS, YTS, _, _ = load_ensemble_data(
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


@mark.svm
@mark.builder
@mark.architect
def test_ensemble_builder_without_data(skope):
    check_skope(skope, "svm")
    ens = BuilderEnsemble()
    assert ens.blueprint == "ensemble"
    assert ens.steps_per_epoch > 0
    ens.load_saved_model(arch="ensemble")
    assert ens.model is not None
    assert len(ens.model.layers) == 28
    assert ens.model_path == "models/ensemble/ensembleSVM"
    ens.find_tx_file()
    assert ens.tx_file == "models/ensemble/tx_data.json"


@mark.cal
@mark.builder
@mark.architect
def test_cal_builder_without_data(skope):
    check_skope(skope, "cal")
    builder = Builder(name="mem_clf")
    builder.load_saved_model("calmodels")
    assert builder.model is not None
    assert builder.blueprint == "mem_clf"
    builder.get_blueprint(builder.blueprint)
    assert len(builder.model.layers) == 8
    assert builder.model_path == "models/calmodels/mem_clf"
    builder.find_tx_file()
    assert builder.tx_file == "models/calmodels/tx_data.json"
