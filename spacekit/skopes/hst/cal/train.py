from argparse import ArgumentParser
from spacekit.datasets import hst_cal
from spacekit.preprocessor.scrub import ScrubCal
from spacekit.builder.networks import (
    MemoryClassifier,
    MemoryRegressor,
    WallclockRegressor,
)
from spacekit.analyzer.compute import ComputeMulti, ComputeRegressor

"""
## to load results from disk in a separate session (for plotting, analysis etc):
# bcom2 = ComputeMulti(res_path=f"{res_path}/mem_bin")
# bin_out = bcom2.upload()
# bcom2.load_results(bin_out)
"""


def load_prep(data_path="data/2021-11-04-1636048291/latest.csv"):
    df = hst_cal.load_data(fpath=data_path)
    data = ScrubCal(df).prep_data()
    return data


def build_fit(BuildClass, data, y_train, y_test, test_idx, model_path=None):
    builder = BuildClass(data.X_train, y_train, data.X_test, y_test, test_idx=test_idx)
    builder.build_mlp()
    builder.fit()
    if model_path:
        builder.save_model(output_path=model_path)
    return builder


def compute_cache(builder, res_path):
    if builder.algorithm == "linreg":
        com = ComputeRegressor(builder=builder, res_path=res_path)
        com.calculate_results()
    elif builder.algorithm == "multiclass":
        com = ComputeMulti(builder=builder, res_path=res_path)
        com.calculate_multi()
    outputs = com.make_outputs()
    print(outputs.keys())
    return com


def train_models(data, res_path, mem_clf=1, mem_reg=1, wall_reg=1, model_path=None):
    model_objects = {}
    if mem_clf:
        y_train, y_test, test_idx = data.y_bin_train, data.y_bin_test, data.bin_test_idx
        clf = build_fit(
            MemoryClassifier, data, y_train, y_test, test_idx, model_path=model_path
        )
        bcom = compute_cache(clf, res_path=f"{res_path}/mem_bin")
        model_objects["mem_clf"] = {"builder": clf, "results": bcom}
    if mem_reg:
        y_train, y_test, test_idx = data.y_mem_train, data.y_mem_test, data.mem_test_idx
        mem = build_fit(
            MemoryRegressor, data, y_train, y_test, test_idx, model_path=model_path
        )
        mcom = compute_cache(mem, res_path=f"{res_path}/memory")
        model_objects["mem_reg"] = {"builder": mem, "results": mcom}
    if wall_reg:
        y_train, y_test, test_idx = (
            data.y_wall_train,
            data.y_wall_test,
            data.wall_test_idx,
        )
        wall = build_fit(
            WallclockRegressor, data, y_train, y_test, test_idx, model_path=model_path
        )
        wcom = compute_cache(wall, res_path=f"{res_path}/wallclock")
        model_objects["wall_reg"] = {"builder": wall, "results": wcom}
    return model_objects


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        default="data/2021-11-04-1636048291",
        help="source data base directory path",
    )
    parser.add_argument(
        "--fname", type=str, default="latest.csv", help="name of training data csv file"
    )
    parser.add_argument(
        "--mem_clf", type=int, default=1, help="bool: train memory bin classifier"
    )
    parser.add_argument(
        "--mem_reg", type=int, default=1, help="bool: train memory regressor"
    )
    parser.add_argument(
        "--wall_reg", type=int, default=1, help="bool: train wallclock regressor"
    )
    parser.add_argument(
        "--save_models",
        type=int,
        default=0,
        help="save trained models and weights to disk",
    )
    args = parser.parse_args()
    # import and preprocess data
    data = load_prep(data_path=f"{args.source_path}/data/{args.fname}")
    res_path = f"{args.source_path}/results"
    if args.save_models is True:
        model_path = args.source_path  # auto creates "models" subdir
    else:
        model_path = None
    model_objects = train_models(
        data,
        res_path,
        mem_clf=args.mem_clf,
        mem_reg=args.mem_reg,
        wall_reg=args.wall_reg,
        model_path=model_path,
    )
