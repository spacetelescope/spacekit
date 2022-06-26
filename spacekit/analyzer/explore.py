# STANDARD libraries
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import subplots
import plotly.offline as pyo
import plotly.figure_factory as ff
from keras.preprocessing import image
from scipy.stats import iqr
from spacekit.preprocessor.transform import PowerX

plt.style.use("seaborn-bright")
font_dict = {"family": '"Titillium Web", monospace', "size": 16}
mpl.rc("font", **font_dict)


class ImagePreviews:
    """Base parent class for rendering and displaying images as plots"""

    def __init__(self, X, y):
        self.X = X
        self.y = y


class SVMPreviews(ImagePreviews):
    """ImagePreviews subclass for previewing SVM images. Primarily can be used to compare original with augmented versions.

    Parameters
    ----------
    ImagePlots : class
        spacekit.analyzer.explore.ImagePreviews parent class
    """

    def __init__(self, X, y, X_prime, y_prime):
        """Instantiates an SVMPreviews class object.

        Parameters
        ----------
        X : ndarray
            ndimensional array of image pixel values
        y : ndarray
            target class labels
        X_prime : ndarray
            ndimensional array of augmented image pixel values
        y_prime : ndarray
            target class labels for the augmented images
        """
        super().__init__(X, y)
        self.X_prime = X_prime
        self.y_prime = y_prime

    def preview_pair(self):
        """Plots an original image with its augmented versions."""
        A = self.X
        B = self.X_prime

        plt.figure(figsize=(10, 10))
        for n in range(3):
            x = image.array_to_img(A[n][0])
            ax = plt.subplot(3, 3, n + 1)
            ax.imshow(x)
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for n in range(3):
            x = image.array_to_img(B[n][0])
            ax = plt.subplot(3, 3, n + 1)
            ax.imshow(x)
            plt.axis("off")
        plt.show()

    def preview_augmented(self):
        """Finds the matching positive class images from both image sets and displays them in a grid."""
        posA = self.X[-self.X_prime.shape[0] :][self.y[-self.X_prime.shape[0] :] == 1]
        posB = self.X_prime[self.y_prime == 1]

        plt.figure(figsize=(10, 10))
        for n in range(5):
            x = image.array_to_img(posA[n][0])
            ax = plt.subplot(5, 5, n + 1)
            ax.imshow(x)
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for n in range(5):
            x = image.array_to_img(posB[n][0])
            ax = plt.subplot(5, 5, n + 1)
            ax.imshow(x)
            plt.axis("off")
        plt.show()


class DataPlots:
    """Parent class for drawing exploratory data analysis plots from a dataframe."""

    def __init__(self, df, width=1300, height=700, show=False, save_html=None):
        self.df = df
        self.width = width
        self.height = height
        self.show = show
        self.save_html = save_html
        self.target = None  # target (y) name e.g. "label", "memory", "wallclock"
        self.labels = None  #
        self.classes = None  # target classes e.g. [0,1] or [0,1,2,3]
        self.n_classes = None
        self.group = None  # e.g. "detector" or "instr"
        self.gkeys = None
        self.categories = None
        self.cmap = ["dodgerblue", "gold", "fuchsia", "lime"]
        self.continuous = None
        self.categorical = None
        self.feature_list = None
        self.telescope = None
        self.figures = None
        self.scatter = None
        self.bar = None
        self.groupedbar = None
        self.kde = None

    def group_keys(self):
        if self.group in ["instr", "instrument"]:
            keys = ["acs", "cos", "stis", "wfc3"]
        elif self.group in ["det", "detector"]:
            uniq = list(self.df[self.group].unique())
            if len(uniq) == 2:
                keys = ["wfc-uvis", "other"]
            else:
                keys = ["hrc", "ir", "sbc", "uvis", "wfc"]
        # TODO: target classification / "category"
        elif self.group in ["cat", "category"]:
            keys = [
                "calibration",
                "galaxy",
                "galaxy_cluster",
                "ISM",
                "star",
                "stellar_cluster",
                "unidentified",
            ]
        # TODO: filters
        group_keys = dict(enumerate(keys))
        return group_keys

    def map_data(self):
        """Instantiates grouped dataframes for each detector

        Returns
        -------
        dict
            data_map dictionary of grouped data frames and color map
        """
        if self.cmap is None:
            cmap = ["#119dff", "salmon", "#66c2a5", "fuchsia", "#f4d365"]
        else:
            cmap = self.cmap
        self.data_map = {}
        for key, name in self.gkeys.items():
            data = self.categories[name]
            self.data_map[name] = dict(data=data, color=cmap[key])
        return self.data_map

    def feature_subset(self):
        """Create a set of groups from a categorical feature (dataframe column). Used for plotting multiple traces on a figure

        Returns
        -------
        dictionary
            self.categories attribute containing key-value pairs: groups of observations (values) for each category (keys)
        """
        self.categories = {}
        feature_groups = self.df.groupby(self.group)
        for i in list(range(len(feature_groups))):
            dx = feature_groups.get_group(i)
            k = self.gkeys[i]
            self.categories[k] = dx
        return self.categories

    def feature_stats_by_target(self, feature):
        """Calculates statistical info (mean and standard deviation) for a feature within each target class.

        Parameters
        ----------
        feature : str
            dataframe column to get statistical calculations on

        Returns
        -------
        nested lists
            list of means and list of standard deviations for a feature, subdivided for each target class.
        """
        means, errs = [], []
        for c in self.classes:
            mu, ste = [], []
            for k in list(self.gkeys.keys()):
                data = self.df[
                    (self.df[self.target] == c) & (self.df[self.group] == k)
                ][feature]
                mu.append(np.mean(data))
                ste.append(np.std(data) / np.sqrt(len(data)))
            means.append(mu)
            errs.append(ste)
        return means, errs

    def make_subplots(self, figtype, xtitle, ytitle, data1, data2, name1, name2):
        fig = subplots.make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(name1, name2),
            shared_yaxes=False,
            x_title=xtitle,
            y_title=ytitle,
        )
        fig.add_trace(data1.data[0], 1, 1)
        fig.add_trace(data1.data[1], 1, 1)
        fig.add_trace(data2.data[0], 1, 2)
        fig.add_trace(data2.data[1], 1, 2)

        fig.update_layout(
            title_text=f"{name1} vs {name2}",
            margin=dict(t=50, l=80),
            width=self.width,
            height=self.height,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={
                "color": "#ffffff",
            },
        )
        if self.show:
            fig.show()
        if self.save_html:
            if not os.path.exists(self.save_html):
                os.makedirs(self.save_html, exist_ok=True)
            pyo.plot(
                fig, filename=f"{self.save_html}/{figtype}_{self.name1}_vs_{self.name2}"
            )
        return fig

    def make_scatter_figs(
        self,
        xaxis_name,
        yaxis_name,
        marker_size=15,
        cmap=["cyan", "fuchsia"],
        categories=None,
        target=None,
    ):
        if categories is None:
            categories = {"all": self.df}
        if target is None:
            target = self.target
        scatter_figs = []
        for key, data in categories.items():
            target_groups = data.groupby(target)
            traces = []
            for i in list(range(len(target_groups))):
                dx = target_groups.get_group(i)
                trace = go.Scatter(
                    x=dx[xaxis_name],
                    y=dx[yaxis_name],
                    text=dx.index,
                    mode="markers",
                    opacity=0.7,
                    marker={"size": marker_size, "color": cmap[i]},
                    name=self.labels[i],  # "aligned",
                )
                traces.append(trace)

            layout = go.Layout(
                xaxis={"title": xaxis_name},
                yaxis={"title": yaxis_name},
                title=key,
                # margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                hovermode="closest",
                paper_bgcolor="#242a44",
                plot_bgcolor="#242a44",
                font={"color": "#ffffff"},
                width=700,
                height=500,
            )
            fig = go.Figure(data=traces, layout=layout)
            if self.show:
                fig.show()
            if self.save_html:
                if not os.path.exists(self.save_html):
                    os.makedirs(self.save_html, exist_ok=True)
                pyo.plot(
                    fig,
                    filename=f"{self.save_html}/{key}-{xaxis_name}-{yaxis_name}-{target}-scatter.html",
                )
            scatter_figs.append(fig)
        return scatter_figs

    def make_target_scatter(self, target=None):
        if target is None:
            target = self.target
        target_figs = {}
        for f in self.feature_list:
            target_figs[f] = self.make_scatter_figs(f, target)
        return target_figs

    def bar_plots(
        self,
        X,
        Y,
        feature,
        y_err=[None, None],
        width=700,
        height=500,
        cmap=["dodgerblue", "fuchsia"],
    ):

        traces = []
        for i in self.classes:
            i = int(i)
            trace = go.Bar(
                x=X,
                y=Y[i],
                error_y=dict(type="data", array=y_err[i], color="white", thickness=0.5),
                name=self.labels[i],
                text=sorted(list(self.group_keys().values())),
                marker=dict(color=cmap[i]),
            )
            traces.append(trace)

        layout = go.Layout(
            title=f"{feature.upper()} average by {self.group.capitalize()}",
            xaxis={"title": self.group},
            yaxis={"title": f"{feature} (mean)"},
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
            width=width,
            height=height,
        )
        fig = go.Figure(data=traces, layout=layout)
        if self.save_html:
            pyo.plot(fig, filename=f"{self.save_html}/{feature}-barplot.html")
        if self.show:
            fig.show()
        else:
            return fig

    def kde_plots(
        self,
        cols,
        norm=False,
        targets=False,
        hist=True,
        curve=True,
        binsize=0.2,  # [0.3, 0.2, 0.1]
        width=700,
        height=500,
        cmap=["#F66095", "#2BCDC1"],
    ):
        if norm is True:
            df = PowerX(self.df, cols=cols, join_data=True).Xt
            cols = [c + "_scl" for c in cols]
            tag = "-norm"
        else:
            df = self.df
            tag = ""
        if targets is True:
            hist_data = [df.loc[df[self.target] == c][cols[0]] for c in self.classes]
            group_labels = self.labels  # [f"{cols[0]}={i}" for i in self.labels]
            title = f"KDE {cols[0]} by target class ({self.target})"
            name = f"kde-targets-{cols[0]}{tag}.html"
        else:
            hist_data = [df[c] for c in cols]
            group_labels = cols
            title = f"KDE {group_labels[0]} vs {group_labels[1]}"
            name = f"kde-{group_labels[0]}-{group_labels[1]}{tag}.html"

        fig = ff.create_distplot(
            hist_data,
            group_labels,
            colors=cmap,
            bin_size=binsize,
            show_hist=hist,
            show_curve=curve,
        )

        fig.update_layout(
            title_text=title,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
            width=width,
            height=height,
        )
        if self.save_html:
            if not os.path.exists(self.save_html):
                os.makedirs(self.save_html, exist_ok=True)
            pyo.plot(fig, filename=f"{self.save_html}/{name}")
        if self.show:
            fig.show()
        return fig

    def scatter3d(self, x, y, z, mask=None, target=None):
        if mask is None:
            df = self.df
        else:
            df = mask
        if target is None:
            target = self.target
        traces = []
        for targ, group in df.groupby(target):
            trace = go.Scatter3d(
                x=group[x],
                y=group[y],
                z=group[z],
                name=targ,
                mode="markers",
                marker=dict(size=7, color=targ, colorscale="Plasma", opacity=0.8),
            )
            traces.append(trace)
        layout = go.Layout(
            title=f"3D Scatterplot: {x} - {y} - {z}",
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
            legend_title_text=target,
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z))
        if self.save_html:
            pyo.plot(fig, filename=f"{self.save_html}/scatter3d.html")
        if self.show:
            fig.show()
        else:
            return fig

    def remove_outliers(self, y_data):
        q = y_data.quantile([0.25, 0.75]).values
        q1, q3 = q[0], q[1]
        lower_fence = q1 - 1.5 * iqr(y_data)
        upper_fence = q3 + 1.5 * iqr(y_data)
        y = y_data.loc[(y_data > lower_fence) & (y_data < upper_fence)]
        return y

    def box_plots(self, cols=None, outliers=True):
        box = {}
        title_sfx = ""
        if cols is None:
            features = self.continuous
        else:
            features = cols
        for f in features:
            traces = []
            for i, name in enumerate(self.gkeys.values()):
                y_data = self.categories[name][f]
                if outliers is False:
                    y_data = self.remove_outliers(y_data)
                    title_sfx = "- no outliers"
                trace = go.Box(y=y_data, name=name, marker=dict(color=self.cmap[i]))
                traces.append(trace)

            layout = go.Layout(
                title=f"{f} by {self.group}{title_sfx}",
                hovermode="closest",
                paper_bgcolor="#242a44",
                plot_bgcolor="#242a44",
                font={"color": "#ffffff"},
            )
            fig = go.Figure(data=traces, layout=layout)
            box[f] = fig
        return box

    def grouped_barplot(self, target="label", cmap=None, save=False):
        df = self.df
        if cmap is None:
            cmap = ["red", "orange", "yellow", "purple", "blue"]
        groups = df.groupby([self.group])[target]
        traces = []
        for key, value in self.gkeys.items():
            dx = groups.get_group(key).value_counts()
            trace = go.Bar(
                x=dx.index, y=dx, name=value.upper(), marker=dict(color=cmap[key])
            )
            traces.append(trace)
        layout = go.Layout(title=f"{target.title()} by {self.group.title()}")
        fig = go.Figure(data=traces, layout=layout)
        if self.save_html:
            pyo.plot(fig, filename=f"{self.save_html}/grouped-bar.html")
        if self.show:
            fig.show()
        else:
            return fig


class HstSvmPlots(DataPlots):
    """Instantiates an HstSvmPlots class

    Parameters
    ----------
    DataPlots : class
        spacekit.analyzer.explore.DataPlots parent class
    """

    def __init__(
        self, df, group="det", width=1300, height=700, show=False, save_html=None
    ):
        super().__init__(df, width=width, height=height, show=show, save_html=save_html)
        self.group = group
        self.telescope = "HST"
        self.target = "label"
        self.classes = list(set(df[self.target].values))  # [0, 1]
        self.labels = ["aligned", "misaligned"]
        self.n_classes = len(set(self.labels))
        self.gkeys = super().group_keys()
        self.categories = self.feature_subset()
        self.continuous = ["rms_ra", "rms_dec", "gaia", "nmatches", "numexp"]
        self.categorical = ["det", "wcs", "cat"]
        self.feature_list = self.continuous + self.categorical
        self.cmap = ["#119dff", "salmon", "#66c2a5", "fuchsia", "#f4d365"]
        self.df_by_detector()
        self.bar = None
        self.scatter = None
        self.kde = None

    def draw_plots(self):
        self.bar = self.alignment_bars()
        self.scatter = self.alignment_scatters()
        self.kde = self.alignment_kde()

    def alignment_bars(self):
        self.bar = {}
        X = sorted(list(self.gkeys.keys()))
        for f in self.continuous:
            means, errs = self.feature_stats_by_target(f)
            bar = self.bar_plots(X, means, f, y_err=errs)
            self.bar[f] = bar
        return self.bar

    def alignment_scatters(self):
        rms_scatter = self.make_scatter_figs(
            "rms_ra", "rms_dec", categories=self.categories
        )
        source_scatter = self.make_scatter_figs(
            "point", "segment", categories=self.categories
        )
        self.scatter = {"rms_ra_dec": rms_scatter, "point_segment": source_scatter}
        return self.scatter

    def alignment_kde(self):
        cols = self.continuous
        self.kde = dict(rms=self.kde_plots(["rms_ra", "rms_dec"]), targ={}, norm={})
        targ = [self.kde_plots([c], targets=True) for c in cols]
        norm = [self.kde_plots([c], norm=True, targets=True) for c in cols]
        for i, c in enumerate(cols):
            self.kde["targ"][c] = targ[i]
            self.kde["norm"][c] = norm[i]
        return self.kde

    # def group_keys(self):
    #     if self.group in ["det", "detector"]:
    #         keys = ["hrc", "ir", "sbc", "uvis", "wfc"]
    #     elif self.group in ["cat", "category"]:
    #         keys = [
    #             "calibration",
    #             "galaxy",
    #             "galaxy_cluster",
    #             "ISM",
    #             "star",
    #             "stellar_cluster",
    #             "unidentified",
    #         ]
    #     group_keys = dict(enumerate(keys))
    #     return group_keys

    def df_by_detector(self):
        """Instantiates grouped dataframes for each detector

        Returns
        -------
        self
        """
        try:
            self.hrc = self.df.groupby("det").get_group(0)
            self.ir = self.df.groupby("det").get_group(1)
            self.sbc = self.df.groupby("det").get_group(2)
            self.uvis = self.df.groupby("det").get_group(3)
            self.wfc = self.df.groupby("det").get_group(4)
            self.instr_dict = {
                "hrc": [self.hrc, "#119dff"],  # lightblue
                "ir": [self.ir, "salmon"],
                "sbc": [self.sbc, "#66c2a5"],  # lightgreen
                "uvis": [self.uvis, "fuchsia"],
                "wfc": [self.wfc, "#f4d365"],  # softgold
            }
        except Exception as e:
            print(e)
        return self


class HstCalPlots(DataPlots):
    def __init__(self, df, group="instr"):
        super().__init__(df)
        self.telescope = "HST"
        self.target = "mem_bin"
        self.classes = [0, 1, 2, 3]
        self.group = group
        self.labels = ["2g", "8g", "16g", "64g"]
        self.gkeys = self.group_keys()
        self.categories = self.feature_subset()
        self.acs = None
        self.cos = None
        self.stis = None
        self.wfc3 = None
        self.instr_dict = None
        self.instruments = list(self.df["instr_key"].unique())
        self.continuous = ["n_files", "total_mb", "x_files", "x_size"]
        self.categorical = [
            "drizcorr",
            "pctecorr",
            "crsplit",
            "subarray",
            "detector",
            "dtype",
            "instr",
        ]
        self.feature_list = self.continuous + self.categorical
        self.cmap = ["dodgerblue", "gold", "fuchsia", "lime"]
        self.data_map = None
        self.scatter = None
        self.box = None
        self.scatter3 = None

    def df_by_instr(self):
        self.acs = self.df.groupby("instr").get_group(0)
        self.cos = self.df.groupby("instr").get_group(1)
        self.stis = self.df.groupby("instr").get_group(2)
        self.wfc3 = self.df.groupby("instr").get_group(3)
        self.instr_dict = {
            "acs": [self.acs, "#119dff"],
            "wfc3": [self.wfc3, "salmon"],
            "cos": [self.cos, "#66c2a5"],
            "stis": [self.stis, "fuchsia"],
        }
        return self

    def draw_plots(self):
        self.scatter = self.make_cal_scatterplots()
        self.box = self.box_plots()
        box_target = self.box_plots(cols=["memory", "wallclock"])
        box_fenced = self.box_plots(cols=["memory", "wallclock"], outliers=False)
        self.box["memory"] = box_target["memory"]
        self.box["wallclock"] = box_target["wallclock"]
        self.box["mem_fence"] = box_fenced["memory"]
        self.box["wall_fence"] = box_fenced["wallclock"]
        # self.scatter3 = self.make_cal_scatter3d()
        # self.bar
        # self.kde

    def make_cal_scatterplots(self):
        memory_figs, wallclock_figs = {}, {}
        for f in self.feature_list:
            memory_figs[f] = self.make_scatter_figs(f, "memory")
            wallclock_figs[f] = self.make_scatter_figs(f, "wallclock")
        self.scatter = dict(memory=memory_figs, wallclock=wallclock_figs)
        return self.scatter

    def make_cal_scatter3d(self):
        x, y = "memory", "wallclock"
        self.scatter3 = {}
        for z in self.continuous:
            data = self.df[[x, y, z, "instr_key"]]
            scat3d = super().scatter3d(
                x, y, z, mask=data, target="instr_key", width=700, height=700
            )
            self.scatter3[z] = scat3d

    def make_box_figs(self, vars):
        box_figs = []
        for v in vars:
            data = [
                go.Box(y=self.acs[v], name="acs"),
                go.Box(y=self.cos[v], name="cos"),
                go.Box(y=self.stis[v], name="stis"),
                go.Box(y=self.wfc3[v], name="wfc3"),
            ]
            layout = go.Layout(
                title=f"{v} by instrument",
                hovermode="closest",
                paper_bgcolor="#242a44",
                plot_bgcolor="#242a44",
                font={"color": "#ffffff"},
            )
            fig = go.Figure(data=data, layout=layout)
            box_figs.append(fig)
        return box_figs

    def make_scatter_figs(self, xaxis_name, yaxis_name):
        if self.data_map is None:
            self.map_data()
        scatter_figs = []
        for instr, datacolor in self.data_map.items():
            data = datacolor["data"]
            color = datacolor["color"]
            trace = go.Scatter(
                x=data[xaxis_name],
                y=data[yaxis_name],
                text=data.index,
                mode="markers",
                opacity=0.7,
                marker={"size": 15, "color": color},
                name=instr,
            )
            layout = go.Layout(
                xaxis={"title": xaxis_name},
                yaxis={"title": yaxis_name},
                title=instr,
                hovermode="closest",
                paper_bgcolor="#242a44",
                plot_bgcolor="#242a44",
                font={"color": "#ffffff"},
            )
            fig = go.Figure(data=trace, layout=layout)
            scatter_figs.append(fig)
        return scatter_figs


class SignalPlots:
    @staticmethod
    def atomic_vector_plotter(
        signal,
        label_col=None,
        classes=None,
        class_names=None,
        figsize=(15, 5),
        y_units=None,
        x_units=None,
    ):
        """
        Plots scatter and line plots of time series signal values.

        **ARGS
        signal: pandas series or numpy array
        label_col: name of the label column if using labeled pandas series
            -use default None for numpy array or unlabeled series.
            -this is simply for customizing plot Title to include classification
        classes: (optional- req labeled data) tuple if binary, array if multiclass
        class_names: tuple or array of strings denoting what the classes mean
        figsize: size of the figures (default = (15,5))

        ******

        Ex1: Labeled timeseries passing 1st row of pandas dataframe
        > first create the signal:
        signal = x_train.iloc[0, :]
        > then plot:
        atomic_vector_plotter(signal, label_col='LABEL',classes=[1,2],
                    class_names=['No Planet', 'Planet']), figsize=(15,5))

        Ex2: numpy array without any labels
        > first create the signal:
        signal = x_train.iloc[0, :]

        >then plot:
        atomic_vector_plotter(signal, figsize=(15,5))
        """
        import pandas as pd
        import numpy as np

        # pass None to label_col if unlabeled data, creates generic title
        if label_col is None:
            label = None
            title_scatter = "Scatterplot of Star Flux Signals"
            title_line = "Line Plot of Star Flux Signals"
            color = "black"

        # store target column as variable
        elif label_col is not None:
            label = signal[label_col]
            # for labeled timeseries
            if label == 1:
                cn = class_names[0]
                color = "red"

            elif label == 2:
                cn = class_names[1]
                color = "blue"
            # TITLES
            # create appropriate title acc to class_names
            title_scatter = f"Scatterplot for Star Flux Signal: {cn}"
            title_line = f"Line Plot for Star Flux Signal: {cn}"

        # Set x and y axis labels according to units
        # if the units are unknown, we will default to "Flux"
        if y_units is None:
            y_units = "Flux"
        else:
            y_units = y_units
        # it is assumed this is a timeseries, default to "time"
        if x_units is None:
            x_units = "Time"
        else:
            x_units = x_units

        # Scatter Plot
        if type(signal) == np.array:
            series_index = list(range(len(signal)))

            converted_array = pd.Series(signal.ravel(), index=series_index)
            signal = converted_array

        plt.figure(figsize=figsize)
        plt.scatter(
            pd.Series([i for i in range(1, len(signal))]),
            signal[1:],
            marker=4,
            color=color,
        )
        plt.ylabel(y_units)
        plt.xlabel(x_units)
        plt.title(title_scatter)
        plt.show()

        # Line Plot
        plt.figure(figsize=figsize)
        plt.plot(pd.Series([i for i in range(1, len(signal))]), signal[1:], color=color)
        plt.ylabel(y_units)
        plt.xlabel(x_units)
        plt.title(title_line)
        plt.show()

    @staticmethod
    def flux_specs(
        signal,
        Fs=2,
        NFFT=256,
        noverlap=128,
        mode="psd",
        cmap=None,
        units=None,
        colorbar=False,
        save_for_ML=False,
        fname=None,
        num=None,
        **kwargs,
    ):
        """generate and save spectographs of flux signal frequencies"""
        import matplotlib.pyplot as plt

        if cmap is None:
            cmap = "binary"

        # PIX: plots only the pixelgrids -ideal for image classification
        if save_for_ML is True:
            # turn off everything except pixel grid
            fig, ax = plt.subplots(figsize=(10, 10), frameon=False)
            fig, freqs, t, m = plt.specgram(
                signal, Fs=Fs, NFFT=NFFT, mode=mode, cmap=cmap
            )
            ax.axis(False)
            ax.show()

            if fname is not None:
                try:
                    if num:
                        path = fname + num
                    else:
                        path = fname
                    plt.savefig(path, **kwargs)
                except Exception as e:
                    print("Something went wrong while saving the img file")
                    print(e)

        else:
            fig, ax = plt.subplots(figsize=(13, 11))
            fig, freqs, t, m = plt.specgram(
                signal, Fs=Fs, NFFT=NFFT, mode=mode, cmap=cmap
            )
            plt.colorbar()
            if units is None:
                units = ["Wavelength (λ)", "Frequency (ν)"]
            plt.xlabel(units[0])
            plt.ylabel(units[1])
            if num:
                title = f"Spectrogram_{num}"
            else:
                title = "Spectrogram"
            plt.title(title)
            plt.show()

        return fig, freqs, t, m

    @staticmethod
    def singal_phase_folder(file_list, fmt="kepler.fits", error=False, snr=False):
        """plots phase-folded light curve of a signal
        returns dataframe of transit timestamps for each light curve
        planet_hunter(f=files[9], fmt='kepler.fits')

        args:
        - fits_files = takes array of files or single .fits file

        kwargs:
        - format : 'kepler.fits' or  'tess.fits'
        - error: include SAP flux error (residuals) if available
        - snr: apply signal-to-noise-ratio to periodogram autopower calculation
        """
        from astropy.timeseries import TimeSeries
        import numpy as np
        from astropy import units as u
        from astropy.timeseries import BoxLeastSquares
        from astropy.stats import sigma_clipped_stats
        from astropy.timeseries import aggregate_downsample

        # read in file
        transits = {}
        for index, file in enumerate(file_list):
            res = {}
            if fmt == "kepler.fits":
                prefix = file.replace("ktwo", "")
                suffix = prefix.replace("_llc.fits", "")
                pair = suffix.split("-")
                obs_id = pair[0]
                campaign = pair[1]

            ts = TimeSeries.read(file, format=fmt)  # read in timeseries

            # add to meta dict
            res["obs_id"] = obs_id
            res["campaign"] = campaign
            res["lc_start"] = ts.time.jd[0]
            res["lc_end"] = ts.time.jd[-1]

            # use box least squares to estimate period
            if error is True:  # if error col data available
                periodogram = BoxLeastSquares.from_timeseries(
                    ts, "sap_flux", "sap_flux_err"
                )
            else:
                periodogram = BoxLeastSquares.from_timeseries(ts, "sap_flux")
            if snr is True:
                results = periodogram.autopower(0.2 * u.day, objective="snr")
            else:
                results = periodogram.autopower(0.2 * u.day)

            maxpower = np.argmax(results.power)
            period = results.period[maxpower]
            transit_time = results.transit_time[maxpower]

            res["maxpower"] = maxpower
            res["period"] = period
            res["transit"] = transit_time

            # res['ts'] = ts

            # fold the time series using the period
            ts_folded = ts.fold(period=period, epoch_time=transit_time)

            # folded time series plot
            # plt.plot(ts_folded.time.jd, ts_folded['sap_flux'], 'k.', markersize=1)
            # plt.xlabel('Time (days)')
            # plt.ylabel('SAP Flux (e-/s)')

            # normalize the flux by sigma-clipping the data to determine the baseline flux:
            mean, median, stddev = sigma_clipped_stats(ts_folded["sap_flux"])
            ts_folded["sap_flux_norm"] = ts_folded["sap_flux"] / median
            res["mean"] = mean
            res["median"] = median
            res["stddev"] = stddev
            res["sap_flux_norm"] = ts_folded["sap_flux_norm"]

            # downsample the time series by binning the points into bins of equal time
            ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.03 * u.day)

            # final result
            fig = plt.figure(figsize=(11, 5))
            ax = fig.gca()
            ax.plot(ts_folded.time.jd, ts_folded["sap_flux_norm"], "k.", markersize=1)
            ax.plot(
                ts_binned.time_bin_start.jd,
                ts_binned["sap_flux_norm"],
                "r-",
                drawstyle="steps-post",
            )
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Normalized flux")
            ax.set_title(obs_id)
            ax.legend([np.round(period, 3)])
            plt.close()

            res["fig"] = fig

            transits[index] = res

        df = pd.DataFrame.from_dict(transits, orient="index")

        return df


# testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path to dataframe (csv file)")
    parser.add_argument("index", type=str, default="index", help="index column name")
    parser.add_argument(
        "-e", "--example", type=str, choices=["svm", "cal"], help="run example demo"
    )
    args = parser.parse_args()
    dataset = args.dataset
    index = args.index
    example = args.example

    df = pd.read_csv(dataset, index_col=index)

    if example == "svm":
        # Drop extra columns in case raw / un-preprocessed dataset is loaded
        drops = ["category", "ra_targ", "dec_targ", "imgname"]
        df.drop([c for c in drops if c in df.columns], axis=1, inplace=True)
        svm = HstSvmPlots(df)
    else:
        print("More examples coming soon!")
