from argparse import ArgumentParser
import os
from spacekit.datasets.hst_cal import calcloud_data, calcloud_uri
from spacekit.extractor.scrape import WebScraper
from spacekit.analyzer.scan import CalScanner, import_dataset
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

def preprocess_data(source_data="data", primary=-1):
    # 1 - get data archives (job data, models, results)
    if source_data == "github":
        _ = WebScraper(calcloud_uri, calcloud_data).scrape_repo()
        data_dir = "data"
    else:
        data_dir = source_data
    # 2 - import data
    cal = CalScanner(perimeter=f"{data_dir}/20??-*-*-*", primary=primary)
    df = import_dataset(
        filename=cal.data,
        kwargs=dict(index_col="ipst"),
        decoder_key={"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}},
    )
     # 3 - data preprocessing
    data = ScrubCal(df).prep_data()
    selection = cal.datapaths[cal.primary]
    res_path = f"{selection}/results"
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
    if mem_clf:
        clf, bCom = train_memory_classifier(data, res_path, save_model_path=mp, save_results=res)
    if mem_reg:
        mem, mCom = train_memory_regressor(data, res_path, save_model_path=mp, save_results=res)
    if wall_reg:
        wall, wCom = train_wallclock_regressor(data, res_path, save_model_path=mp, save_results=res)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_data", type=str, default="data", help="path to dataset directory")
    parser.add_argument("--primary", type=int, default=-1)
    parser.add_argument("--mem_clf", type=int, default=1)
    parser.add_argument("--mem_reg", type=int, default=1)
    parser.add_argument("--wall_reg", type=int, default=1)
    parser.add_argument("--save_model_path", type=str, default=None)
    parser.add_argument("--save_results", type=int, default=1)
    args = parser.parse_args()
    # import and preprocess data
    data, res_path = preprocess_data(source_data=args.source_data, primary=args.primary)
    # train models (save results)
    mp = args.save_model_path
    res = args.save_results
    # clf, bCom = train_memory_classifier(data, res_path, save_model_path=mp, save_results=res)
    # mem, mCom = train_memory_regressor(data, res_path, save_model_path=mp, save_results=res)
    # wall, wCom = train_wallclock_regressor(data, res_path, save_model_path=mp, save_results=res)
    train_models(
        mem_clf=args.mem_clf,
        mem_reg=args.mem_reg,
        wall_reg=args.wall_reg,
        mp=mp,
        res=res
    )
