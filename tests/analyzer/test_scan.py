from pytest import mark


EXPECTED = {
    "svm": {
        "classes": [0, 1],
        "labels": ["aligned", "misaligned"],
        "target": "test",
        "target2": "val",
        "versions": ["v0", "v1"],
        "date": "2022-01-16",
        "time": 1642337739,
        "comtype1": "<class 'spacekit.analyzer.compute.ComputeBinary'>",
        "comtype2": "<class 'spacekit.analyzer.compute.ComputeBinary'>",
        "metric": "acc_loss",
        "score_index": ["train_acc", "train_loss", "test_acc", "test_loss"],
        "score_vals": [1.0, 0.01, 0.92, 0.65],
        "fig_lengths": [2, 2, 4],
        "keras_title": "Accuracy",
        "roc_label": "lab_0",
        "cmx_shape": (2, 2),
        "df_ncol": 12,
        "df_key": "det_key",
    },
    "hstcal": {
        "classes": [0, 1, 2, 3],
        "labels": ["2g", "8g", "16g", "64g"],
        "target": "mem_bin",
        "target2": "wallclock",
        "versions": ["v0", "v1"],
        "date": "2021-10-28",
        "time": 1635457222,
        "comtype1": "<class 'spacekit.analyzer.compute.ComputeMulti'>",
        "comtype2": "<class 'spacekit.analyzer.compute.ComputeRegressor'>",
        "metric": "acc_loss",
        "score_index": ["train_acc", "train_loss", "test_acc", "test_loss"],
        "score_vals": [0.96, 0.08, 0.96, 0.09],
        "fig_lengths": [2, 2, 4],
        "keras_title": "Accuracy",
        "roc_label": "bin_0",
        "cmx_shape": (4, 4),
        "df_ncol": 22,
        "df_key": "instr_key",
    },
}


def exp(scanner):
    global e
    e = EXPECTED[scanner.exp]
    return e


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_attrs(scanner):
    e = exp(scanner)
    assert scanner.classes == e["classes"]
    assert scanner.labels == e["labels"]
    assert scanner.target == e["target"]


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_results(scanner):
    scanner.scan_results()
    assert list(scanner.mega.keys()) == e["versions"]
    v0 = scanner.mega["v0"]
    v1 = scanner.mega["v1"]
    target = e["target"]
    assert v0["date"] == e["date"]
    assert v0["time"] == e["time"]
    assert e["target"] in v0["res"]
    com0 = v0["res"][target]
    com1 = v1["res"][target]
    assert com0 != com1
    assert str(type(com1)) == e["comtype1"]
    target2 = e["target2"]
    com2 = v0["res"][target2]
    assert str(type(com2)) == e["comtype2"]


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_compare_scores(scanner):
    metric = e["metric"]
    scanner.compare_scores(metric=metric)
    assert list(scanner.scores.index) == e["score_index"]
    assert list(scanner.scores.columns) == e["versions"]
    assert list(scanner.scores.v0.values) == e["score_vals"]


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_acc_loss_bars(scanner):
    scanner.acc_fig = scanner.accuracy_bars()
    scanner.loss_fig = scanner.loss_bars()
    scanner.acc_loss_subplots()
    figs = [scanner.acc_fig, scanner.loss_fig, scanner.acc_loss_fig]
    fig_lengths = e["fig_lengths"]
    for fig, length in list(zip(figs, fig_lengths)):
        assert str(type(fig)) == "<class 'plotly.graph_objs._figure.Figure'>"
        assert len(fig.data) == length


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_make_barplots(scanner):
    scanner.scores, scanner.acc_fig, scanner.loss_fig, scanner.acc_loss_fig = (
        None,
        None,
        None,
        None,
    )
    fig_lengths = e["fig_lengths"]
    fig_lengths.extend([len(e["score_vals"])])
    scanner.make_barplots()
    attr_names = ["acc_fig", "loss_fig", "acc_loss_fig", "scores"]
    for attr, length in list(zip(attr_names, fig_lengths)):
        if attr == "scores":
            assert len(scanner.__getattribute__(attr)) == length
        else:
            assert len(scanner.__getattribute__(attr).data) == length


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_make_clf_plots(scanner):
    target = e["target"]
    scanner.make_clf_plots(target=target)
    for v in scanner.versions:
        assert scanner.keras[v][0].layout["title"]["text"] == e["keras_title"]
        assert scanner.roc[v][0].data[0]["name"].split(" ")[0] == e["roc_label"]
    assert list(scanner.cmx.keys()) == ["normalized", "counts"]
    for i in list(range(len(scanner.cmx["normalized"]))):
        assert scanner.cmx["normalized"][i].shape == e["cmx_shape"]
        assert scanner.cmx["counts"][i].shape == e["cmx_shape"]
        assert (
            str(type(scanner.cmx["normalized"][i].ravel()[0]))
            == "<class 'numpy.float64'>"
        )
        assert str(type(scanner.cmx["counts"][i].ravel()[0])) == "<class 'numpy.int64'>"


@mark.hst
@mark.analyzer
@mark.scan
def test_scan_load_dataframe(scanner):
    scanner.load_dataframe()
    assert scanner.df.shape[1] == e["df_ncol"]
    assert len(scanner.df) > 0
    assert e["df_key"] in scanner.df.columns
