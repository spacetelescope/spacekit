from pytest import mark, fixture
from pytest import lazy_fixture

EXPECTED = {
    "svm": {
        "classes": [0, 1],
        "labels": ['aligned', 'misaligned'],
        "target": "test",
        "target2": "val",
        "versions": ['v0', 'v1'],
        "date": '2022-01-16',
        "time": 1642337739,
        "comtype1": "<class 'spacekit.analyzer.compute.ComputeBinary'>",
        "comtype2": "<class 'spacekit.analyzer.compute.ComputeBinary'>",
        "metric": "acc_loss",
        "score_index": ['train_acc', 'train_loss', 'test_acc', 'test_loss'],
        "score_vals": [1.0, 0.01, 0.92, 0.65],
        "fig_lengths": [2, 2, 4],
        "keras_title": "Accuracy",
        "roc_label": "lab_0",
        "cmx_shape": (2, 2),
        "df_ncol": 12,
        "df_key": "det_key"
    },
    "cal": {
        "classes": [0, 1, 2, 3],
        "labels": ['2g', '8g', '16g', '64g'],
        "target": "mem_bin",
        "target2": "wallclock",
        "versions": ['v0', 'v1'],
        "date": '2021-10-28',
        "time": 1635457222,
        "comtype1": "<class 'spacekit.analyzer.compute.ComputeMulti'>",
        "comtype2": "<class 'spacekit.analyzer.compute.ComputeRegressor'>",
        "metric": "acc_loss",
        "score_index": ['train_acc', 'train_loss', 'test_acc', 'test_loss'],
        "score_vals": [0.96, 0.08, 0.96, 0.09],
        "fig_lengths": [2, 2, 4],
        "keras_title": "Accuracy",
        "roc_label": "bin_0",
        "cmx_shape": (4, 4),
        "df_ncol": 22,
        "df_key": "instr_key"
    }
}


@fixture(scope="module") #params=["cal"]
def exp_scan(request):
    return EXPECTED[request.param]


@mark.analyzer
@mark.scan
@mark.parametrize((lazy_fixture("scanner"),lazy_fixture("exp_scan")), [
    ("svm", "svm"),
    ("cal", "cal")
    ])
def test_scan_attrs(scanner, exp_scan):
    assert scanner.classes == exp_scan["classes"]
    assert scanner.labels == exp_scan["labels"]
    assert scanner.target == exp_scan["target"]


@mark.analyzer
@mark.scan
def test_scan_results(scanner, exp_scan):
    scanner.scan_results()
    assert list(scanner.mega.keys()) == exp_scan["versions"]
    v0 = scanner.mega['v0']
    v1 = scanner.mega['v1']
    target = exp_scan["target"]
    assert v0['date'] == exp_scan["date"]
    assert v0['time'] == exp_scan["time"]
    assert exp_scan["target"] in v0['res']
    com0 = v0['res'][target]
    com1 = v1['res'][target]
    assert com0 != com1
    assert str(type(com1)) == exp_scan["comtype1"]
    target2 = exp_scan["target2"]
    com2 = v0['res'][target2]
    assert str(type(com2)) == exp_scan["comtype2"]


@mark.analyzer
@mark.scan
def test_scan_compare_scores(scanner, exp_scan):
    metric = exp_scan["metric"]
    scanner.compare_scores(metric=metric)
    assert list(scanner.scores.index) == exp_scan["score_index"]
    assert list(scanner.scores.columns) ==  exp_scan["versions"]
    assert list(scanner.scores.v0.values) == exp_scan["score_vals"]


@mark.analyzer
@mark.scan
def test_scan_acc_loss_bars(scanner, exp_scan):
    scanner.acc_fig = scanner.accuracy_bars()
    scanner.loss_fig = scanner.loss_bars()
    scanner.acc_loss_subplots()
    figs = [scanner.acc_fig, scanner.loss_fig, scanner.acc_loss_fig]
    fig_lengths = exp_scan["fig_lengths"]
    for fig, length in list(zip(figs, fig_lengths)):
        assert str(type(fig)) == "<class 'plotly.graph_objs._figure.Figure'>"
        assert len(fig.data) == length


@mark.analyzer
@mark.scan
def test_scan_make_barplots(scanner, exp_scan):
    scanner.scores, scanner.acc_fig, scanner.loss_fig, scanner.acc_loss_fig = None, None, None, None
    fig_lengths = exp_scan["fig_lengths"]
    fig_lengths.extend([len(exp_scan["score_vals"])])
    scanner.make_barplots()
    attr_names = ["acc_fig", "loss_fig", "acc_loss_fig", "scores"]
    for attr, length in list(zip(attr_names, fig_lengths)):
        if attr == "scores":
            assert len(scanner.__getattribute__(attr)) == length
        else:
            assert len(scanner.__getattribute__(attr).data) == length


@mark.analyzer
@mark.scan
def test_scan_make_clf_plots(scanner, exp_scan):
    target = exp_scan["target"]
    scanner.make_clf_plots(target=target)
    for v in scanner.versions:
        assert scanner.keras[v][0].layout["title"]["text"] == exp_scan["keras_title"]
        assert scanner.roc[v][0].data[0]["name"].split(" ")[0] == exp_scan["roc_label"]
    assert list(scanner.cmx.keys()) == ["normalized", "counts"]
    for i in list(range(len(scanner.cmx['normalized']))):
        assert scanner.cmx['normalized'][i].shape == exp_scan["cmx_shape"]
        assert scanner.cmx['counts'][i].shape == exp_scan["cmx_shape"]
        assert str(type(scanner.cmx["normalized"][i].ravel()[0])) == "<class 'numpy.float64'>"
        assert str(type(scanner.cmx["counts"][i].ravel()[0])) == "<class 'numpy.int64'>"

@mark.analyzer
@mark.scan
def test_scan_load_dataframe(scanner, exp_scan):
    scanner.load_dataframe()
    assert scanner.df.shape[1] == exp_scan["df_ncol"]
    assert len(scanner.df) > 0
    assert exp_scan["df_key"] in scanner.df.columns
