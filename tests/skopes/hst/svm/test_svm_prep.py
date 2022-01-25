from pytest import mark
import os
from spacekit.skopes.hst.svm.prep import run_preprocessing

@mark.svm
@mark.prep
def test_prep(svm_visit_data):
    fname = run_preprocessing(svm_visit_data, fname="test_prep", output_path="tmp")
    assert os.path.exists(fname)

    img_dir = os.path.join(os.path.dirname(fname), "img")
    visit = os.listdir(img_dir)
    assert len(visit) == 1

    images = os.listdir(os.path.join(img_dir, visit[0]))
    assert len(images) == 3
