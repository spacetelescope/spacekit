from pytest import mark
from spacekit.skopes.hst.svm.train import run_training, load_ensemble_data, train_ensemble, compute_results

params = dict(
    batch_size=32,
    epochs=1,
    lr=1e-4,
    decay=[100000, 0.96],
    early_stopping=None,
    verbose=2,
    ensemble=True,
)

@mark.skip
@mark.svm
@mark.train
@mark.parametrize("norm", [(0), (1)])
def test_svm_training(svm_labeled_dataset, svm_train_img, norm):
    ens, com, val = run_training(
        svm_labeled_dataset, 
        svm_train_img,
        img_size=128,
        norm=norm,
        model_name="ensembleSVM",
        params=params,
        output_path="tmp"
        )
    assert True

# @mark.parametrize("norm", [(0, 1)])
# def test_load_training_data():
#     tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = load_ensemble_data(
#         data_file, svm_train_img, img_size=img_size, norm=norm, output_path=output_path
#     )
#     assert True

# def test_train_svm_model():
#     ens = train_ensemble(
#         XTR,
#         YTR,
#         XTS,
#         YTS,
#         model_name=model_name,
#         params=params,
#         output_path=output_path,
#     )
#     com, val = compute_results(ens, tv_idx, val_set=(XVL, YVL), output_path=output_path)
#     assert True
