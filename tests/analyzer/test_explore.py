import pandas as pd
from spacekit.analyzer.explore import HstSvmPlots
from pytest import parametrize, fixture, lazy_fixture, mark

# @parametrize(["scanner", "exp_scan"], [
#     (lazy_fixture('svm_scanner'), lazy_fixture('svm_exp_scan')),
#     (lazy_fixture('cal_scanner'), lazy_fixture('cal_exp_scan'))
# ])
@fixture(scope='class')
def hst_svm():
    fname = "tests/data/svm/train/training.csv"
    df = pd.read_csv(fname, index_col="index")
    hst = HstSvmPlots(df)
    return hst

@mark.analyzer
@mark.explore
@parametrize(["hst"], (lazy_fixture('hst_svm')))
class TestExploreSvm:

    def make_scatter_plots(hst):
        hst.make_svm_scatterplots()
        assert True
