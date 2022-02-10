from pytest import mark, fixture, lazy_fixture
from spacekit.analyzer.scan import SvmScanner, CalScanner
import plotly.graph_objs


EXPECTED = {
    "svm": {
        "classes": [0, 1],
        "labels": ['aligned', 'misaligned'],
        "target": "test",
        "versions": ['v0', 'v1', 'v2'],
        "date": '2022-01-16',
        "time": 1642337739,
        "metric": "acc_loss",
        "score_index": ['train_acc', 'train_loss', 'test_acc', 'test_loss'],
        "score_vals": [1.0, 0.01, 0.92, 0.65],
        "acc_loss_length": [2, 2, 4],
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
        "versions": ['v0', 'v1', 'v2'],
        "date": '2021-10-28',
        "time": 1635457222,
        "metric": "acc_loss",
        "score_index": ['train_acc', 'train_loss', 'test_acc', 'test_loss'],
        "score_vals": [1.0, 0.01, 0.92, 0.65],
        "fig_lengths": [2, 2, 4],
        "keras_title": "Accuracy",
        "roc_label": "bin_0",
        "cmx_shape": (2, 2),
        "df_ncol": 12,
        "df_key": "instr_key"
    }
}

@fixture(scope='class')
def svm_exp_scan():
    return EXPECTED["svm"]

@fixture(scope='class')
def cal_exp_scan():
    return EXPECTED["cal"]

@fixture(scope='class')
def svm_scanner():
    scanner = SvmScanner(perimeter="data/20??-*-*-*", primary=-1)
    return scanner

@fixture(scope='class')
def cal_scanner():
    scanner = CalScanner(perimeter="data/20??-*-*-*", primary=-1)
    return scanner



# @mark.parametrize("scanner, exp_scan", [
#     (svm_scanner, svm_exp_scan),
#     (cal_scanner, cal_exp_scan)
# ])
@mark.analyzer
@mark.scan
@mark.parametrize(["scanner", "exp_scan"], [
    (lazy_fixture('svm_scanner'), lazy_fixture('svm_exp_scan')),
    (lazy_fixture('cal_scanner'), lazy_fixture('cal_exp_scan'))
])
class TestScanner:

    def test_svm_attrs(self, scanner, exp_scan):
        assert scanner.classes == exp_scan["classes"]
        assert scanner.labels == exp_scan["labels"]
        assert scanner.target == exp_scan["target"]

    def test_svm_scan_results(self, scanner, exp_scan):
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

    def test_svm_compare_scores(self, scanner, exp_scan):
        metric = exp_scan["metric"]
        scanner.compare_scores(metric=metric)
        assert list(scanner.scores.index) == exp_scan["score_index"]
        assert list(scanner.scores.columns) ==  exp_scan["versions"]
        assert list(scanner.scores.v0.values) == exp_scan["score_vals"]

    def test_svm_acc_loss_bars(self, scanner, exp_scan):
        acc_fig = scanner.accuracy_bars()
        loss_fig = scanner.loss_bars()
        acc_loss_fig = scanner.acc_loss_subplots()
        figs = [acc_fig, loss_fig, acc_loss_fig]
        fig_lengths = exp_scan["fig_lengths"]
        for fig, length in list(zip(figs, fig_lengths)):
            assert type(fig) == plotly.graph_objs._figure.Figure
            assert len(fig.data) == length

    def test_svm_make_barplots(self, scanner, exp_scan):
        size = exp_scan["fig_lengths"]
        scanner.make_barplots()
        assert len(scanner.__getattribute__("scores").data) == 4
        assert len(scanner.__getattribute__("acc_fig").data) == size[0]
        assert len(scanner.__getattribute__("loss_fig").data) == size[1]
        assert len(scanner.__getattribute__("acc_loss_fig").data) == size[2]

    def test_svm_make_clf_plots(self, scanner, exp_scan):
        target = exp_scan["target"]
        scanner.make_clf_plots(target=target)
        for v in scanner.versions:
            assert scanner['keras'][v][0].layout["title"]["text"] == exp_scan["keras_title"]
            assert scanner.roc[v][0].data[0]["name"].split(" ")[0] == exp_scan["roc_label"]
        assert list(scanner.cmx.keys()) == ["normalized", "counts"]
        for i in list(range(len(scanner.cmx['normalized']))):
            assert scanner.cmx['normalized'][i].shape == exp_scan["cmx_shape"]
            assert scanner.cmx['counts'][i].shape == exp_scan["cmx_shape"]

    def test_svm_load_dataframe(self, scanner, exp_scan):
        scanner.load_dataframe()
        assert scanner.df.shape[1] == exp_scan["df_ncols"]
        assert len(scanner.df) > 0
        assert exp_scan["df_key"] in scanner.df.columns

# @mark.analyzer
# @mark.scan
# class TestSvmScanner:

#     def test_svm_attrs(svm_scanner):
#         assert svm_scanner.classes == [0, 1]
#         assert svm_scanner.labels == ['aligned', 'misaligned']
#         assert svm_scanner.target == 'test'

#     def test_svm_scan_results(svm_scanner):
#         svm_scanner.scan_results()
#         assert list(svm_scanner.mega.keys()) == ['v0', 'v1', 'v2']
#         v0 = svm_scanner.mega['v0']
#         v1 = svm_scanner.mega['v1']
#         assert v0['date'] == '2022-01-16'
#         assert v0['time'] == 1642337739
#         assert 'test' in v0['res']
#         tcom0 = v0['res']['test']
#         tcom1 = v1['res']['test']
#         assert tcom0 != tcom1

#     def test_svm_compare_scores(svm_scanner):
#         svm_scanner.compare_scores(metric='acc_loss')
#         assert list(svm_scanner.scores.index) == [
#             'train_acc', 'train_loss', 'test_acc', 'test_loss'
#         ]
#         assert list(svm_scanner.scores.columns) ==  ['v0', 'v1', 'v2']
#         assert list(svm_scanner.scores.v0.values) == [1.0, 0.01, 0.92, 0.65]

#     def test_svm_acc_loss_bars(svm_scanner):
#         acc_fig = svm_scanner.accuracy_bars()
#         loss_fig = svm_scanner.loss_bars()
#         acc_loss_fig = svm_scanner.acc_loss_subplots()
#         figs = [acc_fig, loss_fig, acc_loss_fig]
#         for fig in figs:
#             assert type(fig) == plotly.graph_objs._figure.Figure
#             assert len(fig.data) >= 2

#     def test_svm_make_barplots(svm_scanner):
#         svm_scanner.make_barplots()
#         assert len(svm_scanner.__getattribute__("scores").data) == 4
#         assert len(svm_scanner.__getattribute__("acc_fig").data) == 2
#         assert len(svm_scanner.__getattribute__("loss_fig").data) == 2
#         assert len(svm_scanner.__getattribute__("acc_loss_fig").data) == 4

#     def test_svm_make_clf_plots(svm_scanner):
#         svm_scanner.make_clf_plots(target="test")
#         for v in svm_scanner.versions:
#             assert svm_scanner['keras'][v][0].layout["title"]["text"] == "Accuracy"
#             assert svm_scanner.roc[v][0].data[0]["name"].split(" ")[0] == 'lab_0'
#         assert list(svm_scanner.cmx.keys()) == ["normalized", "counts"]
#         for i in list(range(len(svm_scanner.cmx['normalized']))):
#             assert svm_scanner.cmx['normalized'][i].shape == (2, 2)
#             assert svm_scanner.cmx['counts'][i].shape == (2, 2)

#     def test_svm_load_dataframe(svm_scanner):
#         svm_scanner.load_dataframe()
#         assert svm_scanner.df.shape[1] == 12
#         assert len(svm_scanner.df) > 0
#         assert 'det_key' in svm_scanner.df.columns
