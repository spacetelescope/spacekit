import os
from pytest import mark
from spacekit.skopes.hst.svm.prep import run_preprocessing
import pandas as pd

EXPECTED_PREP = ["test_prep.csv", "test_prep.h5", "raw_test_prep.csv"]

@mark.svm
@mark.prep
def test_run_prep(single_visit_path):
    df, fname = run_preprocessing(single_visit_path, fname="test_prep", output_path="tmp")
    assert len(df) > 0
    assert os.path.exists(fname)
    img_dir = os.path.join(os.path.dirname(fname), "img")
    visit = os.listdir(img_dir)
    assert len(visit) == 1
    images = os.listdir(os.path.join(img_dir, visit[0]))
    assert len(images) == 3
    df = pd.read_csv(fname, index_col="index")
    assert len(df) > 0
    for f in EXPECTED_PREP:
        assert f in os.listdir("tmp")

#TODO: test_load_h5