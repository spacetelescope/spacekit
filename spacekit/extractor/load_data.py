import pandas as pd
import pickle
import datetime as dt

# timestamps = [1620351000, 1620740441, 1620929899, 1621096666]
# versions = ["v0", "v1", "v2", "v3"]

# metadata = {
#     "v0": {"date": "2021-05-06-1620351000", "files": res_files},
#     "v1": {"date": "2021-05-11-1620740441", "files": res_files},
#     "v2": {"date": "2021-05-13-1620929899", "files": res_files},
#     "v3": {"date": "2021-05-15-1621096666", "files": res_files}
# }


def format_res_files():
    clf_files = [
        "history",
        #"kfold",
        "matrix",
        "preds",
        "proba",
        "scores",
        "y_pred",
        "y_true",
    ]
    reg_files = ["history", "predictions", "residuals", "scores"]
    res_files = {"mem_bin": clf_files, "memory": reg_files, "wallclock": reg_files}
    return res_files


def make_meta(timestamps, versions):
    res_files = format_res_files()
    meta = {}
    for v in versions:
        meta[v] = {"date": "", "files": res_files}
    for (v, t) in list(zip(versions, timestamps)):
        DATE = dt.date.fromtimestamp(t).isoformat()
        datestring = f"{DATE}-{t}"
        meta[v] = {"date": datestring, "files": res_files}
    return meta


def load_res_file(meta, metric, version, target):
    """Locates pickle obj for a given metric, loads and stores into variable
    Returns res_file (loaded pickle object).
    Ex: keras_file = load_res_file("history", "v0", "mem_bin")
    loads from: "./data/2021-05-06-1620351000/results/mem_bin/history"
    """
    datestring = meta[version]["date"]
    res_file = f"data/{datestring}/results/{target}/{metric}"
    # print(f"{metric} file located: \n{res_file}")
    res_data = pickle.load(open(res_file, "rb"))
    return res_data


def make_res(meta, versions=None):
    if versions is None:
        versions = list(meta.keys())
    results = {}
    for v in versions:
        results[v] = {"mem_bin": {}, "memory": {}, "wallclock": {}}
        targets = list(meta[v]["files"].keys())
        for t in targets:
            for r in meta[v]["files"][t]:
                results[v][t][r] = load_res_file(meta, r, v, t)
    return results


def get_scores(results):
    df_list = []
    for v in results.keys():
        score_dict = results[v]["mem_bin"]["scores"]
        df = pd.DataFrame.from_dict(score_dict, orient="index", columns=[v])
        df_list.append(df)
    df_scores = pd.concat([d for d in df_list], axis=1)
    return df_scores


def get_history(results, version):
    return results[version]["mem_bin"]["history"]


def get_pred_proba(results, version):
    y_true = results[version]["mem_bin"]["y_true"]
    y_pred = results[version]["mem_bin"]["y_pred"]
    y_proba = results[version]["mem_bin"]["proba"]
    return y_true, y_pred, y_proba


def dynamodb_data(key):
    # TODO: download from ddb
    csv_file = key
    return csv_file


def s3_data(key):
    # TODO: download from s3
    csv_file = key
    return csv_file


def import_csv(src, key, index=None):
    if src == "s3":
        csv_file = s3_data(key)
    elif src == "ddb":
        csv_file = dynamodb_data(key)
    elif src == "file":
        csv_file = key
    # import csv file
    if index:
        df = pd.read_csv(csv_file, index_col=index)
    else:
        df = pd.read_csv(csv_file)
    return df


def get_instruments(df):
    instrument_key = {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}
    for i, name in instrument_key.items():
        df.loc[df["instr"] == i, "instr_key"] = name
    return df

# timestamps = [1629663047, 1635457222, 1636048291]
# versions = ["2021-08-22", "2021-10-28", "2021-11-04"]
# dataset = f"{versions[-1]}-{timestamps[-1]}/data/latest.csv"
# def split_df_by_instrument_timestamp(data, meta, instr):
#     dates = [meta[i]["date"] for i, _ in meta.items()]
#     df = data.loc[data['instr'] == instr]
#     df0 = df.loc[df["timestamp"] == dates[0]]
#     df1 = df.loc[df["timestamp"] == dates[1]]
#     df2 = df.loc[df["timestamp"] == dates[2]]


def get_training_data(meta):
    dates = [meta[i]["date"] for i, _ in meta.items()]
    batches = [f"./data/{d}/batch.csv" for d in dates]
    dataframes = [pd.read_csv(d) for d in batches]
    training_list = [get_instruments(df) for df in dataframes]
    instruments = list(training_list[0]["instr_key"].unique())
    versions = list(meta.keys())
    for dataset, version, date in list(zip(training_list, versions, dates)):
        dataset["version"] = version
        dataset["training_date"] = date
    training_data = pd.concat([d for d in training_list], verify_integrity=False)
    return training_data, instruments


def get_single_dataset(filename):
    data = pd.read_csv(filename)
    data.set_index("ipst", drop=False, inplace=True)
    df = get_instruments(data)  # adds instrument label (string)
    return df
