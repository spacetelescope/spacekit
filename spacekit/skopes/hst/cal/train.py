from argparse import ArgumentParser
import os
from spacekit.datasets import hst_cal
from spacekit.preprocessor.scrub import ScrubCal
from spacekit.builder.networks import (
    MemoryClassifier,
    MemoryRegressor,
    WallclockRegressor,
)
from spacekit.analyzer.compute import ComputeMulti, ComputeRegressor

"""
## to load results from disk in a separate session (for plotting, etc):
# bcom2 = ComputeMulti(res_path=f"{res_path}/mem_bin")
# bin_out = bcom2.upload()
# bcom2.load_results(bin_out)
"""

def load_and_prep(source_dir="data", dataset="2021-11-04-1636048291", fname="latest.csv"):
    # 1 - import data
    base_dir = os.path.join(source_dir, dataset)
    fpath = os.path.join(base_dir, "data", fname)
    df = hst_cal.load_data(fpath=fpath)
     # 2 - data preprocessing
    data = ScrubCal(df).prep_data()
    res_path = f"{base_dir}/results"
    return data, res_path

def train_memory_classifier(data, res_path, save_model_path=None, save_results=True):
    # 4 - build & train models; compute results
    # 4a: Memory Bin Classifier
    clf = MemoryClassifier(
        data.X_train,
        data.y_bin_train,
        data.X_test,
        data.y_bin_test,
        test_idx=data.bin_test_idx,
    )
    clf.build_mlp()
    clf.fit()
    if save_model_path is not None:
        clf.save_model(output_path=save_model_path)
    bCom = ComputeMulti(builder=clf, res_path=f"{res_path}/mem_bin")
    bCom.calculate_multi()
    if save_results is True:
        _ = bCom.make_outputs()
    return clf, bCom

def train_memory_regressor(data, res_path, save_model_path=None, save_results=True):
    # 4b: Memory Regressor
    mem = MemoryRegressor(
        data.X_train,
        data.y_mem_train,
        data.X_test,
        data.y_mem_test,
        test_idx=data.mem_test_idx,
    )
    mem.build_mlp()
    mem.fit()  # using default fit params
    if save_model_path is not None:
        mem.save_model(output_path=save_model_path)
    mCom = ComputeRegressor(builder=mem, res_path=f"{res_path}/memory")
    mCom.calculate_results()
    if save_results is True:
        _ = mCom.make_outputs()
    return mem, mCom


def train_wallclock_regressor(data, res_path, save_model_path=None, save_results=True):
    # 4c: Wallclock Regressor
    wall = WallclockRegressor(
        data.X_train,
        data.y_wall_train,
        data.X_test,
        data.y_wall_test,
        test_idx=data.wall_test_idx,
    )
    wall.build_mlp()
    wall.fit_params(batch_size=64, epochs=300) #, ensemble=False)
    wall.fit()
    if save_model_path is not None:
        wall.save_model(output_path=save_model_path)
    wCom = ComputeRegressor(builder=wall, res_path=f"{res_path}/wallclock")
    wCom.calculate_results()
    if save_results is True:
        _ = wCom.make_outputs()
    return wall, wCom

def train_models(mem_clf=1, mem_reg=1, wall_reg=1, mp=None, res=1):
    model_objects = {}
    if mem_clf:
        clf, bCom = train_memory_classifier(data, res_path, save_model_path=mp, save_results=res)
        model_objects["mem_clf"] = {"model": clf, "results": bCom}
    if mem_reg:
        mem, mCom = train_memory_regressor(data, res_path, save_model_path=mp, save_results=res)
        model_objects["mem_reg"] = {"model": mem, "results": mCom}
    if wall_reg:
        wall, wCom = train_wallclock_regressor(data, res_path, save_model_path=mp, save_results=res)
        model_objects["wall_reg"] = {"model": wall, "results": wCom}
    return model_objects


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data", help="source data base directory path")
    parser.add_argument("--dataset", type=str, default="2021-11-04-1636048291", help="training data timestamp (and subdirectory name)")
    parser.add_argument("--fname", type=str, default="latest.csv", help="name of training data csv file")
    parser.add_argument("--mem_clf", type=int, default=1, help="bool: train memory bin classifier")
    parser.add_argument("--mem_reg", type=int, default=1, help="bool: train memory regressor")
    parser.add_argument("--wall_reg", type=int, default=1, help="bool: train wallclock regressor")
    parser.add_argument("--save_models", type=str, default=0, help="save trained models and weights to disk")
    parser.add_argument("--save_results", type=int, default=1, help="bool: save training results metrics to disk")
    args = parser.parse_args()
    # import and preprocess data
    data, res_path = load_and_prep(source_dir=args.base_path, dataset=args.dataset, fname=args.fname)
    model_objects = train_models(
        mem_clf=args.mem_clf,
        mem_reg=args.mem_reg,
        wall_reg=args.wall_reg,
        mp=args.save_model_path,
        res=args.save_results
    )
