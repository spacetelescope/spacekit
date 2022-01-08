"""
Classes and methods primarily used by spacekit.dashboard but can easily be repurposed.
"""
import os
import pandas as pd
import glob
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import subplots
from spacekit.analyzer.compute import ComputeBinary, ComputeMulti, ComputeRegressor


def decode_categorical(df, decoder_key):
    """Add decoded column (using "{column}_key" suffix) to dataframe.

    Parameters:
        df (Dataframe) :
        decoder_key (Dictionary): key-value pairs of encoding integers and strings
            Ex. (decoder_key)
            instrument_key = {"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}
            detector_key = {"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}}

    Returns:
        df (Pandas Dataframe): dataframe with additional categorical column (object dtype) based on encoding pairs.
    """
    for key, pairs in decoder_key.items():
        for i, name in pairs.items():
            df.loc[df[key] == i, f"{key}_key"] = name
    return df


def import_dataset(filename=None, kwargs=dict(index_col="ipst"), decoder_key=None):
    """Imports and loads dataset from csv file. Optionally decodes an encoded feature back into strings.

    Parameters:
        filename (path, optional) path to csv file. Defaults to None.
        kwargs (dictionary, optional): dict of keyword args to pass into pandas read_csv method.
            Ex: to set the index_col attribute: kwargs=dict(index_col="ipst")
        decoder_key (dict, optional): nested dict of column and key value pairs for decoding a categorical feature into strings.
            Ex: {"instr": {{0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}}

    Returns:
        df (Pandas dataframe): dataframe of imported csv file
    """
    if not os.path.exists(filename):
        print("File could not be found")
    # load dataset
    df = pd.read_csv(filename, **kwargs)
    if decoder_key:
        df = decode_categorical(df, decoder_key)  # adds instrument label (string)
    return df


class MegaScanner:
    def __init__(self, perimeter="data/20??-*-*-*", primary=-1):
        self.perimeter = perimeter
        self.datapaths = sorted(list(glob.glob(perimeter)))
        self.datasets = [d.split("/")[-1] for d in self.datapaths]
        self.timestamps = [
            int(t.split("-")[-1]) for t in self.datasets
        ]  # [1636048291, 1635457222, 1629663047]
        self.dates = [
            str(v)[:10] for v in self.datasets
        ]  # ["2021-11-04", "2021-10-28", "2021-08-22"]
        self.primary = primary
        self.data = None
        self.versions = None
        self.res_keys = None
        self.mega = None
        self.dfs = []
        self.scores = None  # self.compare_scores()
        self.acc_fig = None  # self.acc_bars()
        self.loss_fig = None  # self.loss_bars()
        self.acc_loss_figs = None  # self.acc_loss_subplots()
        self.res_fig = None  # TODO

    def select_dataset(self, primary=None):
        if primary:
            self.primary = primary
        if self.primary > len(self.datapaths):
            print("Using default index (-1)")
            self.primary = -1
            raise IndexError
        if len(self.datapaths) > 0:
            dataset_path = self.datapaths[self.primary]
            self.data = glob.glob(f"{dataset_path}/data/*.csv")[0]
            return self.data
        else:
            return None

    def make_mega(self):
        self.mega = {}
        versions = []
        for i, (d, t) in enumerate(zip(self.dates, self.timestamps)):
            if self.versions is None:
                v = f"v{str(i)}"
                versions.append(v)
            else:
                v = self.versions[i]
            self.mega[v] = {"date": d, "time": t, "res": self.res_keys}
        if len(versions) > 0:
            self.versions = versions
        return self.mega

    def compare_scores(self, target="mem_bin", score_type="acc_loss"):
        df_list = []
        for v in self.versions:
            score_dict = self.mega[v]["res"][target][score_type]
            df = pd.DataFrame.from_dict(score_dict, orient="index", columns=[v])
            df_list.append(df)
        self.scores = pd.concat([d for d in df_list], axis=1)
        return self.scores

    def accuracy_bars(self):
        acc_train = self.scores.loc["train_acc"].values
        acc_test = self.scores.loc["test_acc"].values
        xvals = [c for c in self.scores.columns]
        data = [
            go.Bar(
                x=list(range(len(acc_train))),
                hovertext=xvals,
                y=acc_train,
                name="Training Accuracy",
                marker=dict(color="#119dff"),
            ),
            go.Bar(
                x=list(range(len(acc_test))),
                hovertext=xvals,
                y=acc_test,
                name="Test Accuracy",
                marker=dict(color="#66c2a5"),
            ),
        ]
        layout = go.Layout(
            title="Accuracy",
            xaxis={"title": "training iteration"},
            yaxis={"title": "score"},
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def loss_bars(self):
        loss_train = self.scores.loc["train_loss"].values
        loss_test = self.scores.loc["test_loss"].values
        xvals = [c for c in self.scores.columns]
        data = [
            go.Bar(
                x=list(range(len(loss_train))),
                y=loss_train,
                hovertext=xvals,
                name="Training Loss",
                marker=dict(color="salmon"),
            ),
            go.Bar(
                x=list(range(len(loss_test))),
                y=loss_test,
                hovertext=xvals,
                name="Test Loss",
                marker=dict(color="peachpuff"),
            ),
        ]
        layout = go.Layout(
            title="Loss",
            xaxis={"title": "training iteration"},
            yaxis={"title": "score"},
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(
            data=data,
            layout=layout,
        )
        return fig

    def acc_loss_subplots(self):
        self.acc_loss_fig = subplots.make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Accuracy", "Loss"),
            shared_yaxes=False,
            x_title="Training Iteration",
            y_title="Score",
        )
        self.acc_loss_fig.add_trace(self.acc.data[0], 1, 1)
        self.acc_loss_fig.add_trace(self.acc.data[1], 1, 1)
        self.acc_loss_fig.add_trace(self.loss.data[0], 1, 2)
        self.acc_loss_fig.add_trace(self.loss.data[1], 1, 2)
        self.acc_loss_fig.update_layout(
            title_text="Accuracy vs. Loss",
            margin=dict(t=50, l=200),
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={
                "color": "#ffffff",
            },
        )
        return self.acc_loss_fig

    # REFACTOR there is a definitely a better way to do this
    def single_cmx(self, cmx, zmin, zmax, classes, subtitles=("v0")):
        x = classes
        y = x[::-1].copy()
        z = cmx[::-1]
        z_text = [[str(y) for y in x] for x in z]
        subplot_titles = subtitles

        fig = subplots.make_subplots(
            rows=1,
            cols=1,
            subplot_titles=subplot_titles,
            shared_yaxes=False,
            x_title="Predicted",
            y_title="Actual",
        )
        fig.update_layout(
            title_text="Confusion Matrix",
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        # make traces
        fig1 = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=z_text,
            colorscale="Blues",
            zmin=zmin,
            zmax=zmax,
        )
        fig.add_trace(fig1.data[0], 1, 1)
        annot1 = list(fig1.layout.annotations)
        annos = [annot1]

        # add colorbar
        fig["data"][0]["showscale"] = True
        # annotation values for each square
        for anno in annos:
            fig.add_annotation(anno)
        return fig

    def triple_cmx(self, coms, cmx_type):
        if cmx_type == "normalized":
            zmin = 0.0
            zmax = 1.0
            cmx0, cmx1, cmx2 = coms[0].cmx_norm, coms[1].cmx_norm, coms[2].cmx_norm
        else:
            zmin = 0
            zmax = 100
            cmx0, cmx1, cmx2 = coms[0].cmx, coms[1].cmx, coms[2].cmx
        cmx = [cmx0, cmx1, cmx2]
        classes = coms[0].classes  # ["2GB", "8GB", "16GB", "64GB"]
        x = classes
        y = x[::-1].copy()
        subplot_titles = self.versions  # ("v1", "v2", "v3")
        fig = subplots.make_subplots(
            rows=1,
            cols=3,
            subplot_titles=subplot_titles,
            shared_yaxes=False,
            x_title="Predicted",
            y_title="Actual",
        )
        fig.update_layout(
            title_text="Confusion Matrix",
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        annos = []
        for i in cmx:
            col = i + 1
            z = cmx[i][::-1]
            z_text = [[str(y) for y in x] for x in z]
            cmx_fig = ff.create_annotated_heatmap(
                z=z,
                x=x,
                y=y,
                annotation_text=z_text,
                colorscale="Blues",
                zmin=zmin,
                zmax=zmax,
            )
            fig.add_trace(cmx_fig.data[0], 1, col)
            annot = list(cmx_fig.layout.annotations)

            for k in range(len(annot)):
                annot[k]["xref"] = f"x{str(col)}"
                annot[k]["yref"] = f"y{str(col)}"
            annos.append(annot)
        new_annotations = []
        for a in annos:
            new_annotations.extend(a)
        # add colorbar
        fig["data"][0]["showscale"] = True
        # annotation values for each square
        for anno in new_annotations:
            fig.add_annotation(anno)


class CalScanner(MegaScanner):
    def __init__(self, perimeter="data/20??-*-*-*", primary=-1):
        super().__init__(perimeter=perimeter, primary=primary)
        self.classes = ["2g", "8g", "16g", "64g"]
        self.res_keys = {"mem_bin": {}, "memory": {}, "wallclock": {}}
        self.data = self.select_dataset()
        self.mega = self.make_mega()
        self.scores = None  # self.compare_scores()
        self.acc_fig = None  # self.acc_bars()
        self.loss_fig = None  # self.loss_bars()
        self.acc_loss_figs = None  # self.acc_loss_subplots()

    def scan_results(self):
        self.mega = self.make_mega()
        for i, d in enumerate(self.datapaths):
            v = self.versions[i]
            bCom = ComputeMulti(
                algorithm="clf", classes=self.classes, res_path=f"{d}/results/mem_bin"
            )
            bin_out = bCom.upload()
            bCom.load_results(bin_out)
            self.mega[v]["res"]["mem_bin"] = bCom

            mCom = ComputeRegressor(algorithm="reg", res_path=f"{d}/results/memory")
            mem_out = mCom.upload()
            mCom.load_results(mem_out)
            self.mega[v]["res"]["memory"] = mCom

            wCom = ComputeRegressor(algorithm="reg", res_path=f"{d}/results/wallclock")
            wall_out = wCom.upload()
            wCom.load_results(wall_out)
            self.mega[v]["res"]["wallclock"] = wCom
        return self.mega


class SvmScanner(MegaScanner):
    def __init__(self, perimeter="data/20??-*-*-*", primary=-1):
        super().__init__(perimeter=perimeter, primary=primary)
        self.classes = ["aligned", "misaligned"]
        self.res_keys = {"test": {}, "val": {}}
        self.data = self.select_dataset()
        self.mega = self.make_mega()
        self.scores = None  # self.compare_scores()
        self.acc_fig = None  # self.acc_bars()
        self.loss_fig = None  # self.loss_bars()
        self.acc_loss_figs = None  # self.acc_loss_subplots()

    def scan_results(self):
        self.mega = self.make_mega()
        for i, d in enumerate(self.datasets):
            v = self.versions[i]
            tCom = ComputeBinary(
                algorithm="clf", classes=self.classes, res_path=f"{d}/results/test"
            )
            outputs = tCom.upload()
            tCom.load_results(outputs)
            self.mega[v]["res"]["test"] = tCom

            vCom = ComputeBinary(
                algorithm="clf",
                classes=self.classes,
                res_path=f"{d}/results/val",
                validation=True,
            )
            outputs = vCom.upload()
            vCom.load_results(outputs)
            self.mega[v]["res"]["val"] = vCom
