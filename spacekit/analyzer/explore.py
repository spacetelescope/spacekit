import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import iqr
from spacekit.preprocessor.transform import PowerX
from spacekit.generator.augment import augment_image
from spacekit.logger.log import Logger

try:
    from keras.preprocessing.image import array_to_img
except ImportError:
    from tensorflow.keras.utils import array_to_img

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    font_dict = {"family": "monospace", "size": 16}
    mpl.rc("font", **font_dict)
    styles = ["seaborn-bright", "seaborn-v0_8-bright"]
    valid_styles = [s for s in styles if s in plt.style.available]
    if len(valid_styles) > 0:
        try:
            plt.style.use(valid_styles[0])
        except OSError:
            pass
except ImportError:
    mpl = None
    plt = None

try:
    import plotly.graph_objects as go
    from plotly import subplots
    import plotly.offline as pyo
    import plotly.figure_factory as ff
    import plotly.express as px
except ImportError:
    go = None
    subplots = None
    pyo = None
    ff = None
    px = None


try:
    from astropy.timeseries import TimeSeries, BoxLeastSquares, aggregate_downsample
    from astropy import units as u
    from astropy.stats import sigma_clipped_stats
    from astropy.io import fits
except ImportError:
    TimeSeries = None


def check_ast_imports():
    return TimeSeries is not None


def check_viz_imports():
    return go is not None


def check_mpl_imports():
    return mpl is not None and plt is not None


class ImagePreviews:
    """Base parent class for rendering and displaying images as plots"""

    def __init__(self, X, labels, name="ImagePreviews", **log_kws):
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.X = X
        self.y = labels
        if not check_viz_imports():
            self.log.error("plotly and/or matplotlib not installed.")
            raise ImportError(
                "You must install plotly (`pip install plotly`) "
                "and matplotlib<4 (`pip install matplotlib<4`) "
                "for the compute module to work."
                "\n\nInstall extra deps via `pip install spacekit[x]`"
            )


class SVMPreviews(ImagePreviews):
    """ImagePreviews subclass for previewing SVM images. Primarily can be used to compare original with augmented versions.

    Parameters
    ----------
    ImagePlots : class
        spacekit.analyzer.explore.ImagePreviews parent class
    """

    def __init__(
        self,
        X,
        labels=None,
        names=None,
        ndims=3,
        channels=3,
        w=128,
        h=128,
        figsize=(10, 10),
        **log_kws,
    ):
        """Instantiates an SVMPreviews class object.

        Parameters
        ----------
        X : ndarray
            ndimensional array of image pixel values
        labels : ndarray, optional
            target class labels for each image
        ndims : int, optional
            number of dimensions (frames) per image, by default 3
        channels : int, optional
            channels per image frame (rgb color is 3, gray/bw is 1), by default 3
        w : int, optional
            width of images, by default 128
        h : int, optional
            height of images, by default 128
        """
        super().__init__(X, labels, name="SVMPreviews", **log_kws)
        self.names = names
        self.n_images = len(X)
        self.ndims = ndims
        self.channels = channels
        self.w = w
        self.h = h
        self.figsize = figsize

    def select_image_from_array(self, i=None):
        if i is None:
            return self.X
        else:
            return self.X[i]

    def check_dimensions(self, Xi):
        if Xi.shape != (self.ndims, self.w, self.h, self.channels):
            try:
                Xi = Xi.reshape(self.ndims, self.w, self.h, self.channels)
                return Xi
            except Exception as e:
                print(e)

    def preview_image(self, Xi, dim=3, aug=False, show=False):
        if aug is True:
            # reshape handled by augment if needed
            Xi = augment_image(Xi)
            title = "Augmented"
        else:
            Xi = self.check_dimensions(Xi)
            title = "Original"

        frames = ["orig", "pt-seg", "gaia"]
        fig = px.imshow(
            Xi,
            facet_col=0,
            binary_string=True,
            labels={"facet_col": "frame"},
            facet_col_wrap=3,
        )

        for i, frame in enumerate(frames):
            fig.layout.annotations[i]["text"] = "%s" % frame

        fig.update_layout(
            title_text=f"{title} Image Slices",
            margin=dict(t=100),
            width=990,
            height=500,
            showlegend=False,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={
                "color": "#ffffff",
            },
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        if show is True:
            fig.show()
        else:
            return fig

    def preview_image_mpl(self, Xi, dim=3, aug=False, show=False):
        if aug is True:
            # reshape handled by augment if needed
            Xi = augment_image(Xi)
        else:
            Xi = self.check_dimensions(Xi)

        fig = plt.figure(figsize=self.figsize)
        for n in range(dim):
            xi = array_to_img(Xi[n])
            # xi = image.array_to_img(Xi[n])
            ax = plt.subplot(dim, dim, n + 1)
            ax.imshow(xi)
            plt.axis("off")
        if show is True:
            plt.show()
        else:
            plt.close()
            return fig

    def get_synthetic_image(self, img_name, show=False, dim=3, aug=False):
        pairs = [i for i in self.names if img_name in i]
        if len(pairs) > 1:
            synth_name = pairs[np.argmax([len(p.split("_")) for p in pairs])]
            synth_num = np.where(self.names == synth_name)
            synth_img = self.select_image_from_array(synth_num)
            if show is True:
                self.preview_image(synth_img, dim=dim, aug=aug)
            return synth_name, synth_num, synth_img
        else:
            print("Synthetic version not found for the selected image")
            return None

    def preview_og_aug_pair(self, i=None, dim=3):
        """Plot frames of both original and augmented versions of n-dimensional images

        Parameters
        ----------
        i : int, optional
            index of image selected from array X, by default None
        dim : int, optional
              dimensions (number of frames per image), by default 3
        """
        Xi = self.select_image_from_array(i=i)
        self.preview_image(Xi, dim=dim, aug=False)
        self.preview_image(Xi, dim=dim, aug=True)

    def preview_og_syn_pair(self, img_name):
        pairs = [i for i in self.X if img_name in i]
        self.preview_image(pairs[0])
        self.preview_image(pairs[1])

    # def preview_corrupted_pairs(self):
    #     """Finds the matching positive class images from both image sets and displays them in a grid."""
    #     posA = self.X[-self.X_prime.shape[0] :][self.y[-self.X_prime.shape[0] :] == 1]
    #     posB = self.X_prime[self.y_prime == 1]

    #     plt.figure(figsize=(10, 10))
    #     for n in range(5):
    #         x = image.array_to_img(posA[n][0])
    #         ax = plt.subplot(5, 5, n + 1)
    #         ax.imshow(x)
    #         plt.axis("off")
    #     plt.show()

    #     plt.figure(figsize=(10, 10))
    #     for n in range(5):
    #         x = image.array_to_img(posB[n][0])
    #         ax = plt.subplot(5, 5, n + 1)
    #         ax.imshow(x)
    #         plt.axis("off")
    #     plt.show()


class DataPlots:
    """Base class for drawing exploratory data analysis plots from a dataframe."""

    def __init__(
        self,
        df,
        width=1300,
        height=700,
        show=False,
        save_html=None,
        telescope=None,
        name="DataPlots",
        **log_kws,
    ):
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.df = df
        self.width = width
        self.height = height
        self.show = show
        self.save_html = save_html
        self.telescope = telescope
        self.target = None  # target (y) name e.g. "label", "memory", "wallclock"
        self.labels = None  #
        self.classes = None  # target classes e.g. [0,1] or [0,1,2,3]
        self.n_classes = None
        self.group = None  # e.g. "detector", "instr", "cat"
        self.gkeys = None
        self.group_dict = None
        self.categories = None
        self.cmap = ["dodgerblue", "gold", "fuchsia", "lime"]
        self.continuous = None
        self.categorical = None
        self.feature_list = None
        self.figures = None
        self.scatter = None
        self.bar = None
        self.box = None
        self.groupedbar = None
        self.kde = None
        if not check_viz_imports():
            self.log.error("plotly and/or matplotlib not installed.")
            raise ImportError(
                "You must install plotly (`pip install plotly`) "
                "and matplotlib<4 (`pip install matplotlib<4`) "
                "for the compute module to work."
                "\n\nInstall extra deps via `pip install spacekit[x]`"
            )

    def group_keys(self):
        """Generates numerically ordered key-pairs for each unique value of self.group found in the dataframe

        Returns
        -------
        dict
            enumerated dictionary of unique values for each group
        """
        if not self.group:
            self.log.error(
                "Cannot generate group keys if no grouping feature specified. Set the `group` attribute then try again."
            )
        if self.group.startswith("instr"):
            return dict(enumerate(self.instr_keys()))
        elif self.group.startswith("det"):
            return dict(enumerate(self.det_keys()))
        elif self.group.startswith("cat"):
            return dict(enumerate(self.targ_class_keys()))
        else:
            return dict(enumerate(sorted(list(self.df[self.group].unique()))))

    def instr_keys(self):
        """Generates a list of intruments based on self.telescope

        Returns
        -------
        list
            list of instrument keys for the specified telescope
        """
        if self.telescope not in ["hst", "jwst"]:
            return []
        return dict(hst=["acs", "wfc3", "cos", "stis"], jwst=["fgs", "miri", "nircam", "niriss", "nirspec"])[
            self.telescope.lower()
        ]

    def det_keys(self):
        """Creates a list of detectors based on self.telescope

        Returns
        -------
        list
            list of detector keys for the specified telescope
        """
        keys = sorted(list(self.df[self.group].unique()))
        if self.telescope.lower() == "hst":
            if len(keys) == 2:
                return ["wfc-uvis", "other"]
            if not isinstance(keys[0], str) and len(keys) == 5:
                return ["hrc", "ir", "sbc", "uvis", "wfc"]
        return keys

    def targ_class_keys(self):
        """List of standard astronomical target classification categories

        Returns
        -------
        list
            standard target classification categories
        """
        return [
            "calibration",
            "galaxy",
            "galaxy_cluster",
            "ISM",
            "star",
            "stellar_cluster",
            "unidentified",
        ]

    def map_df_by_group(self):
        """Instantiates `group_dict` as a dictionary of grouped dataframes and color map"""
        self.group_dict = {}
        for k, v in self.gkeys.items():
            self.group_dict[v] = [self.df.groupby(self.group).get_group(k), self.cmap[k]]

    def map_data(self):
        """Instantiates `data_map` as a dictionary of grouped dataframes and color maps for each category in `categories` attribute."""
        cmap = ["#119dff", "salmon", "#66c2a5", "fuchsia", "#f4d365"] if self.cmap is None else self.cmap
        if not self.categories:
            self.feature_subset()
        self.data_map = {}
        for key, name in self.gkeys.items():
            data = self.categories[name]
            self.data_map[name] = dict(data=data, color=cmap[key])

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
            k = self.gkeys[i]
            self.categories[k] = feature_groups.get_group(i)

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
                data = self.df[(self.df[self.target] == c) & (self.df[self.group] == k)][feature]
                mu.append(np.mean(data))
                ste.append(np.std(data) / np.sqrt(len(data)))
            means.append(mu)
            errs.append(ste)
        return means, errs

    def make_subplots(self, figtype, xtitle, ytitle, data1, data2, name1, name2):
        """Generates figure with multiple subplots for two sets of data using previously generated figures.

        Parameters
        ----------
        figtype : str
            type of figure being generated (used for saving html file)
        xtitle : str
            title for the x-axis
        ytitle : str
            title for the y-axis
        data1 : go.Figure
            figure object for the first set of data
        data2 : go.Figure
            figure object for the second set of data
        name1 : str
            name for the first subplot
        name2 : str
            name for the second subplot

        Returns
        -------
        go.Figure
            figure object containing the subplots
        """
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
            pyo.plot(fig, filename=f"{self.save_html}/{figtype}_{self.name1}_vs_{self.name2}")
        return fig

    def make_target_scatter_figs(
        self,
        xaxis_name,
        yaxis_name,
        marker_size=15,
        cmap=["cyan", "fuchsia"],
        categories=None,
        target=None,
    ):
        """Generates scatterplots for two features in the dataframe, grouped by target classes.

        Parameters
        ----------
        xaxis_name : str
            column name in dataframe to plot on x-axis
        yaxis_name : str
            column name in dataframe to plot on y-axis
        marker_size : int, optional
            marker size for scatter plot points, by default 15
        cmap : list, optional
            list of colors for different target classes, by default ["cyan", "fuchsia"]
        categories : dict, optional
            dictionary of categories to group data by, by default None
        target : str, optional
            name of target column in dataframe, by default None
        Returns
        -------
        list
            list of scatterplot figures for each category
        """
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

    def make_feature_scatter_figs(self, xaxis_name, yaxis_name):
        """Generates scatterplots for two features in the dataframe, grouped by the `group` attribute.

        Parameters
        ----------
        xaxis_name : str
            name of column in dataframe to plot on x-axis
        yaxis_name : str
            name of column in dataframe to plot on y-axis

        Returns
        -------
        list
            scatterplot figures for each group in self.group attribute
        """
        if self.data_map is None:
            self.map_data()
        scatter_figs = []
        for key, datacolor in self.data_map.items():
            data = datacolor["data"]
            color = datacolor["color"]
            trace = go.Scatter(
                x=data[xaxis_name],
                y=data[yaxis_name],
                text=data.index,
                mode="markers",
                opacity=0.7,
                marker={"size": 15, "color": color},
                name=key,
            )
            layout = go.Layout(
                xaxis={"title": xaxis_name},
                yaxis={"title": yaxis_name},
                title=key,
                hovermode="closest",
                paper_bgcolor="#242a44",
                plot_bgcolor="#242a44",
                font={"color": "#ffffff"},
            )
            fig = go.Figure(data=trace, layout=layout)
            scatter_figs.append(fig)
        return scatter_figs

    def make_target_scatter(self, target=None):
        """Generates target vs feature scatterplot for a given target (by default self.target) for each feature in self.feature_list.

        Parameters
        ----------
        target : str, optional
            target column name, by default None

        Returns
        -------
        list
            target-feature scatterplot figures for each feature in self.feature_list
        """
        if target is None:
            target = self.target
        target_figs = {}
        for f in self.feature_list:
            target_figs[f] = self.make_target_scatter_figs(f, target)
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
        """Draws a bar plot for a feature, grouped by the `group` attribute.

        Parameters
        ----------
        X : array-like
            X-axis values
        Y : array-like
            Y-axis values
        feature : str
            Feature name
        y_err : list, optional
            Y-axis error values, by default [None, None]
        width : int, optional
            Width of the plot, by default 700
        height : int, optional
            Height of the plot, by default 500
        cmap : list, optional
            List of colors for the plot, by default ["dodgerblue", "fuchsia"]

        Returns
        -------
        go.Figure
            Plotly Figure object representing the bar plot
        """
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
        """Generates KDE plots for specified columns in the dataframe.

        Parameters
        ----------
        cols : list of str
            List of column names to generate KDE plots for
        norm : bool, optional
            Whether to normalize the data, by default False
        targets : bool, optional
            Whether to group data by target classes, by default False
        hist : bool, optional
            Whether to show histogram, by default True
        curve : bool, optional
            Whether to show KDE curve, by default True
        binsize : float, optional
            Bin size for the histogram, by default 0.2
        height : int, optional
            Height of the plot, by default 500
        cmap : list, optional
            List of colors for the plot, by default ["#F66095", "#2BCDC1"]

        Returns
        -------
        go.Figure
            Plotly Figure object representing the KDE plot
        """
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
        """Generates a 3D scatterplot for three features in the dataframe.

        Parameters
        ----------
        x : str
            feature column name for x-axis
        y : str
            feature column name for y-axis
        z : str
            feature column name for z-axis
        mask : pd.DataFrame, optional
            DataFrame to use as a mask/filter, by default None
        target : str, optional
            target column name, by default None

        Returns
        -------
        go.Figure
            Plotly Figure object representing the 3D scatterplot
        """
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
        """Removes outliers from a given pandas Series using the IQR method.

        Parameters
        ----------
        y_data : pd.Series
            The data from which to remove outliers.

        Returns
        -------
        pd.Series
            The data with outliers removed via IQR filtering.
        """
        q = y_data.quantile([0.25, 0.75]).values
        q1, q3 = q[0], q[1]
        lower_fence = q1 - 1.5 * iqr(y_data)
        upper_fence = q3 + 1.5 * iqr(y_data)
        y = y_data.loc[(y_data > lower_fence) & (y_data < upper_fence)]
        return y

    def box_plots(self, cols=None, outliers=True):
        """Generates multi-trace box plots for each feature in cols param, with or without outliers

        Parameters
        ----------
        cols : list, optional
            features to plot from dataframe, by default None (uses self.continuous attribute)
        outliers : bool, optional
            whether to include outliers in the box plots, by default True

        Returns
        -------
        dict
            dictionary of plotly box plot figures for each feature in cols parameter
        """
        box = {}
        title_sfx = ""
        features = cols or self.continuous
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

    def make_box_figs(self, vars: list):
        """Generates single trace box plots, one plot for each var where `vars` is a list of columns in df

        Parameters
        ----------
        vars : list
            column names in dataframe to plot

        Returns
        -------
        list
            list of plotly box plot figures for each variable in vars parameter
        """
        box_figs = []
        if not self.group_dict:
            self.map_df_by_group()
        for v in vars:
            data = [go.Box(y=j[0][v], name=i) for i, j in self.group_dict.items()]
            layout = go.Layout(
                title=f"{v} by {self.group}",
                hovermode="closest",
                paper_bgcolor="#242a44",
                plot_bgcolor="#242a44",
                font={"color": "#ffffff"},
            )
            fig = go.Figure(data=data, layout=layout)
            box_figs.append(fig)
        return box_figs

    def grouped_barplot(self, target="label", cmap=None):
        """Draws a grouped bar plot for a target column, grouped by the `group` attribute.

        Parameters
        ----------
        target : str, optional
            target column to plot, by default "label"
        cmap : list, optional
            list of colors for the bars, by default None

        Returns
        -------
        go.Figure
            plotly figure object for the grouped bar plot
        """
        df = self.df
        if cmap is None:
            cmap = self.cmap or ["red", "orange", "yellow", "purple", "blue"]
        groups = df.groupby([self.group])[target]
        traces = []
        for key, value in self.gkeys.items():
            dx = groups.get_group(key).value_counts()
            trace = go.Bar(x=dx.index, y=dx, name=value.upper(), marker=dict(color=cmap[key]))
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

    def __init__(self, df, group="det", width=1300, height=700, show=False, save_html=None, **log_kws):
        super().__init__(
            df,
            width=width,
            height=height,
            show=show,
            save_html=save_html,
            telescope="hst",
            name="HstSvmPlots",
            **log_kws,
        )
        self.group = group
        self.target = "label"
        self.classes = list(set(df[self.target].values))  # [0, 1]
        self.labels = ["aligned", "misaligned"]
        self.n_classes = len(set(self.labels))
        self.gkeys = self.group_keys()
        self.cmap = ["#119dff", "salmon", "#66c2a5", "fuchsia", "#f4d365"]
        self.feature_subset()
        self.continuous = ["rms_ra", "rms_dec", "gaia", "nmatches", "numexp"]
        self.categorical = ["det", "wcs", "cat"]
        self.feature_list = self.continuous + self.categorical
        self.map_df_by_group()

    def draw_plots(self):
        self.alignment_bars()
        self.alignment_scatters()
        self.alignment_kde()

    def alignment_bars(self):
        self.bar = {}
        X = sorted(list(self.gkeys.keys()))
        for f in self.continuous:
            means, errs = self.feature_stats_by_target(f)
            bar = self.bar_plots(X, means, f, y_err=errs)
            self.bar[f] = bar

    def alignment_scatters(self):
        rms_scatter = self.make_target_scatter_figs("rms_ra", "rms_dec", categories=self.categories)
        source_scatter = self.make_target_scatter_figs("point", "segment", categories=self.categories)
        self.scatter = {"rms_ra_dec": rms_scatter, "point_segment": source_scatter}

    def alignment_kde(self):
        cols = self.continuous
        self.kde = dict(rms=self.kde_plots(["rms_ra", "rms_dec"]), targ={}, norm={})
        targ = [self.kde_plots([c], targets=True) for c in cols]
        norm = [self.kde_plots([c], norm=True, targets=True) for c in cols]
        for i, c in enumerate(cols):
            self.kde["targ"][c] = targ[i]
            self.kde["norm"][c] = norm[i]


class HstCalPlots(DataPlots):
    def __init__(self, df, group="instr", width=1300, height=700, show=False, save_html=None, **log_kws):
        super().__init__(
            df,
            width=width,
            height=height,
            show=show,
            save_html=save_html,
            telescope="hst",
            name="HstCalPlots",
            **log_kws,
        )
        self.target = "mem_bin"
        self.classes = [0, 1, 2, 3]
        self.group = group
        self.labels = ["2g", "8g", "16g", "64g"]
        self.gkeys = self.group_keys()
        self.group_dict = {}
        self.cmap = ["dodgerblue", "gold", "fuchsia", "lime"]
        self.data_map = None
        self.feature_subset()
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
        self.scatter3 = None

    def draw_plots(self):
        self.make_cal_scatterplots()
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
            memory_figs[f] = self.make_feature_scatter_figs(f, "memory")
            wallclock_figs[f] = self.make_feature_scatter_figs(f, "wallclock")
        self.scatter = dict(memory=memory_figs, wallclock=wallclock_figs)

    def make_cal_scatter3d(self):
        x, y = "memory", "wallclock"
        self.scatter3 = {}
        for z in self.continuous:
            data = self.df[[x, y, z, "instr_key"]]
            self.scatter3[z] = super().scatter3d(x, y, z, mask=data, target="instr_key", width=700, height=700)


class SignalPlots:
    """Class for plotting time series signals and their spectrograms."""

    def __init__(
        self,
        show=False,
        save_png=False,
        target_cns={},
        color_map={},
        output_dir=None,
        name="SignalPlots",
        **log_kws,
    ):
        """Class for manipulating and plotting time series signals and frequency spectrograms.

        Parameters
        ----------
        show : bool, optional
            display plot, by default False
        save_png : str, optional
            save plot as PNG file, by default False
        target_cns : dict, optional
            target label and string keypairs, by default {}
        color_map : dict, optional
            target label and color keypairs, by default {}
        """
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.show = show
        self.save_png = save_png
        self.target_cns = target_cns
        self.color_map = color_map
        self.flux_col = "pdcsap_flux"
        self.extra_cols = ["lc_start", "lc_end", "maxpower", "transit", "mean", "median", "stddev"]
        self.output_dir = os.getcwd() if output_dir is None else output_dir
        self.check_dependencies()
        warnings.filterwarnings(action="ignore")  # ignore astropy warnings

    def check_dependencies(self):
        if not check_ast_imports() or not check_mpl_imports():
            self.log.error("astropy and/or matplotlib not installed.")
            raise ImportError(
                "You must have astropy and matplotlib installed "
                f"for the {self.__name__} class to work."
                "\n\nInstall extra deps via `pip install spacekit[x]`"
            )

    def parse_filename(self, fname, fmt="kepler.fits"):
        """Extracts target information from FITS light curve file name.

        Parameters
        ----------
        fname : str
            path to FITS light curve file (llc or lc)
        fmt : str, optional
            'kepler.fits' or 'tess.fits', by default "kepler.fits"

        Returns
        -------
        tuple
            target id (str), campaign/sector id (str)
        """
        fname = os.path.basename(fname)
        if fmt == "kepler.fits":  # r"ktwo{obs_id}-c{campaign}_llc.fits"
            patt = r"ktwo(\d{9,15})-c(\d{2})_llc\.fits"
            m = re.match(patt, fname)
            if m:
                return (m.group(1), m.group(2))  # tid, campaign
        elif fmt == "tess.fits":  # r"tess{date-time}-s{sctr}-{tid}-{scid}-{cr}_lc.fits"
            patt = r"^tess(\d{13})-s(\d{4})-(\d{16,20})-(\d{4})-s_lc\.fits$"
            m = re.match(patt, fname)
            if m:
                return (m.group(2), m.group(1))  # tid, sector
        else:
            raise ValueError("fmt must be 'kepler.fits' or 'tess.fits'")
        raise ValueError("Filename does not match expected pattern")

    @staticmethod
    def read_ts_signal(fits_file, signal_col="pdcsap_flux", fmt="kepler.fits", offset=False, remove_nans=True):
        """Reads time series signal data from a FITS light curve file (_llc.fits or _lc.fits for kepler and fits respectively). Optionally can
        apply telescope-specific BJD offset as determined by `fmt` kwarg (most light curve files already have this applied) and remove NaN values from both signal and corresponding timestamp arrays. Regarding the `signal_col` defaults: "sap_flux" is Simple Aperture Photometry flux, the flux after summing the calibrated pixels within the telescope's optimal photometric aperture; the default (recommended) is "pdcsap_flux" (Pre-search Data Conditioned Simple Aperture Photometry, the SAP flux values nominally corrected for instrumental variations - these are the mission's best estimate of the intrinsic variability of the target.).

        Parameters
        ----------
        fits_file : str
            path to FITS light curve file (llc or lc)
        signal_col : str, optional
            header column name containing the data, by default "pdcsap_flux"
        fmt : str, optional
            'kepler.fits' or 'tess.fits', by default "kepler.fits"
        offset : bool, optional
            apply telescope-specifc BJD offset to timestamps, by default False
        remove_nans : bool, optional
            remove NaN values from signal and timestamps, by default True

        Returns
        -------
        np.ndarray
            time series signal data as a numpy array
        """
        if fmt not in ["kepler.fits", "tess.fits"]:
            raise ValueError("fmt must be 'kepler.fits' or 'tess.fits'")
        ts = TimeSeries.read(fits_file, format=fmt)
        flux = np.asarray(ts[signal_col], dtype="float64")
        timestamps = ts.time.jd
        if offset is True:
            bjd = dict(kepler=2454833.0, tess=2457000.0)[fmt.split(".")[0]]
            timestamps -= bjd  # convert to KBJD/TBJD
        if remove_nans is True:
            not_nan_mask = ~np.isnan(flux)
            flux = flux[not_nan_mask]
            timestamps = timestamps[not_nan_mask]
        return timestamps, flux

    def atomic_vector_plotter(
        self,
        signal,
        timestamps=None,
        label=None,
        y_units="PDCSAP Flux (e-/s)",  # aperture photometry flux
        x_units="Time (BJD)",  # Barycentric Julian Date
        figsize=(15, 10),
        fname="flux_signal.png",
        title_pfx="Flux Signal",
    ):
        """Plots scatter and line plots of time series signal values.

        Parameters
        ----------
        signal : np.ndarray or pandas Series
            time series signal data
        y_units : str, optional
            y-axis label, by default "PDCSAP Flux (e-/s)"
        x_units : str, optional
            x-axis label, by default "Time (BJD)"
        """
        cn = self.target_cns.get(label, "")
        color = self.color_map.get(label, "black")
        title = title_pfx + f": {cn}" if cn != "" else title_pfx
        if timestamps is None:
            timestamps = list(range(len(signal)))
            x_units = "Time Cadence Index"
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
        axs[0].plot(
            timestamps,
            signal,
            color=color,
        )
        axs[0].set_ylabel(y_units)
        axs[1].scatter(
            timestamps,
            signal,
            marker=4,
            color=color,
        )
        axs[1].set_ylabel(y_units)
        plt.xlabel(x_units)
        plt.suptitle(title)
        fig.tight_layout()
        if self.save_png:
            fpath = str(os.path.join(self.output_dir, fname)) + ".png"
            fig.savefig(fpath, dpi=300)
        if self.show:
            plt.show()
        else:
            plt.close()

    def signal_phase_folder(self, file_list, fmt="kepler.fits", error=True, snr=True, include_extra=False):
        """Generates phase-folded light curves from LLC/LCF flux signals

        Parameters
        ----------
        file_list : list
            list of FITS file path(s) containing time series data
        flux_col : str, optional
            header column name containing the data, by default "pdcsap_flux"
        fmt : str, optional
            'kepler.fits' or  'tess.fits', by default "kepler.fits"
        error : bool, optional
            include SAP flux error (residuals) if available, by default True
        snr : bool, optional
            apply signal-to-noise-ratio to periodogram autopower calculation, by default True

        Returns
        -------
        pd.DataFrame
            transit timestamps and phase folded flux values for each light curve
        """
        # req_cols = ["obs_id", "campaign", "time_jd", "sap_flux_norm", "time_bin_start", "sap_flux_norm_binned", "period"]
        transits = {}
        for index, file in enumerate(file_list):
            res = {}
            fname = os.path.basename(file)
            (tid, sc) = self.parse_filename(fname, fmt=fmt)
            ts = TimeSeries.read(file, format=fmt)  # read in timeseries
            # add to meta dict
            res["tid"] = tid
            res["sc"] = sc
            # use box least squares to estimate period
            if error is True and f"{self.flux_col}_err" in ts.columns:
                periodogram = BoxLeastSquares.from_timeseries(ts, self.flux_col, f"{self.flux_col}_err")
            else:
                periodogram = BoxLeastSquares.from_timeseries(ts, self.flux_col)
            if snr is True:
                results = periodogram.autopower(0.2 * u.day, objective="snr")
            else:
                results = periodogram.autopower(0.2 * u.day)
            maxpower = np.argmax(results.power)
            period = results.period[maxpower]
            res["period"] = period
            transit_time = results.transit_time[maxpower]
            # fold the time series using the period
            ts_folded = ts.fold(period=period, epoch_time=transit_time)
            res["time_jd"] = ts_folded.time.jd
            # normalize the flux by sigma-clipping the data to determine the baseline flux:
            mean, median, stddev = sigma_clipped_stats(ts_folded[self.flux_col])
            ts_folded["flux_norm"] = ts_folded[self.flux_col] / median
            res["flux_norm"] = ts_folded["flux_norm"]
            # downsample the time series by binning the points into bins of equal time
            ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.03 * u.day)
            res["time_bin_start"] = ts_binned.time_bin_start.jd
            res["flux_norm_binned"] = ts_binned["flux_norm"]
            if include_extra:
                res["lc_start"] = ts.time.jd[0]
                res["lc_end"] = ts.time.jd[-1]
                res["transit"] = transit_time
                res["maxpower"] = maxpower
                res["mean"] = mean
                res["median"] = median
                res["stddev"] = stddev
                res["fname"] = fname
            transits[index] = res
        df = pd.DataFrame.from_dict(transits, orient="index")
        return df

    def plot_phase_signals(self, ts, title_pfx="Phase-folded Light Curve: ", figsize=(11, 5)):
        """Plots a phase-folded light curve from timeseries flux signal data. Requires a dataframe row containing the following columns:
        "time_jd", "flux_norm", "time_bin_start", "flux_norm_binned", "tid", "sc", "period"
        e.g.,
        df = SignalPlots.signal_phase_folder(file_list)
        ts = df.iloc[index]
        signal_plots.plot_phase_signals(ts)

        Parameters
        ----------
        ts : ArrayLike
            timeseries flux signal data
        title_pfx : str, optional
            Plot title prefix, by default "Phase-folded Light Curve: "
        figsize : tuple, optional
            figure size, by default (11,5)
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.plot(ts["time_jd"], ts["flux_norm"], "k.", markersize=1)
        ax.plot(
            ts["time_bin_start"],
            ts["flux_norm_binned"],
            "r-",
            drawstyle="steps-post",
        )
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(title_pfx + ts["tid"])
        ax.legend([np.round(ts["period"], 3)])
        if self.save_png:
            fpath = os.path.join(self.output_dir, f"{ts['sc']}-{ts['tid']}_phase_folded.png")
            fig.savefig(fpath, dpi=300)
        if self.show:
            plt.show()
        else:
            plt.close()

    def set_spec_kwargs(self, Fs=2, NFFT=256, noverlap=128, mode="psd", cmap="binary"):
        """returns dict of default spectrogram kwargs

        Returns
        -------
        dict
            default spectrogram kwargs
        """
        spec_kwargs = {
            "Fs": Fs,
            "NFFT": NFFT,
            "noverlap": noverlap,
            "mode": mode,
            "cmap": cmap,
        }
        return spec_kwargs

    def flux_specs(
        self,
        signal,
        units=["Wavelength (λ)", "Frequency (ν)"],
        colorbar=True,
        save_for_ml=False,
        fname="specgram",
        title="Spectrogram",
        **kwargs,
    ):
        """generate and save spectrograms of flux signal frequencies. By default uses kwargs in `set_spec_kwargs` method.

        Parameters
        ----------
        signal : ArrayLike
            1D array-like signal data
        units : list of strings, optional
            x and y units respectively, by default N["Wavelength (λ)", "Frequency (ν)"]
        colorbar : bool, optional
            include colorbar in plot, by default True
        save_for_ml : bool, optional
            plots pixel grid only (no axes, colorbar or labels), by default False
        fname : str, optional
            filename without extension for saving png, by default 'specgram'
        title : str, optional
            plot title, by default "Spectrogram"
        **kwargs : dict
            matplotlib.pyplot.specgram keyword arguments

        Returns
        -------
        tuple
            periodogram, freqs, t, m - see matplotlib.pyplot.specgram
        """
        fpath = os.path.join(self.output_dir, fname)
        spec_kwargs = self.set_spec_kwargs(**kwargs)
        if save_for_ml is True:
            fig, ax = plt.subplots(figsize=(10, 10), frameon=False)
            ax.axis(False)
        else:
            fig, ax = plt.subplots(figsize=(13, 11))
            if colorbar:
                plt.colorbar()
            units = ["Wavelength (λ)", "Frequency (ν)"] if units is None or len(units) < 2 else units
            plt.xlabel(units[0])
            plt.ylabel(units[1])
            plt.title(title)

        fig, freqs, t, m = plt.specgram(
            signal,
            **spec_kwargs,
        )
        if self.save_png:
            plt.savefig(fpath, dpi=300)
        if self.show:
            plt.show()
        else:
            plt.close()
        return fig, freqs, t, m


class K2SignalPlots(SignalPlots):
    """Class for plotting K2 time series signals and their spectrograms."""

    def __init__(
        self,
        flux_col="pdcsap_flux",
        show=False,
        save_png=True,
        target_cns={1: "No Planet", 2: "Planet"},
        color_map={1: "red", 2: "blue"},
        **log_kws,
    ):
        """_summary_

        Parameters
        ----------
        show : bool, optional
            display plot, by default False
        save_png : bool, optional
            save plot as PNG file, by default True
        target_cns : dict, optional
            target label and string keypairs, by default {1: "No Planet", 2: "Planet"}
        color_map : dict, optional
            target label and color keypairs, by default {1: "red", 2: "blue"}
        """
        super().__init__(
            show=show,
            save_png=save_png,
            flux_col=flux_col,
            target_cns=target_cns,
            color_map=color_map,
            name="K2SignalPlots",
            **log_kws,
        )
        self.df = None
        self.files = []

    def generate_dataframe(self):
        """Generates dataframe of K2 light curve signal properties from list of FITS files"""
        if len(self.files) == 0:
            raise ValueError("No files provided. Set `self.files` to a list of K2 FITS light curve file paths.")
        self.df = self.signal_phase_folder(self.files, fmt="kepler.fits", error=True, snr=True, include_extra=True)

    def generate_raw_flux_df(self, flux_col="SAP_FLUX", add_label=None, ffillna=True):
        """Generates dataframe of raw flux signals from list of K2 FITS files"""
        if len(self.files) == 0:
            raise ValueError("No files provided. Set `self.files` to a list of K2 FITS light curve file paths.")
        records = {}
        for index, file in enumerate(self.files):
            with fits.open(file) as hdulist:
                signal = hdulist[1].data[flux_col]
                records[index] = np.asarray(signal, dtype="float64")
        df = pd.DataFrame.from_dict(records, orient="index")
        if ffillna is True:
            df.ffill(axis=1, inplace=True)
        df.columns = ["FLUX." + str(c + 1) for c in df.columns]
        if isinstance(add_label, int):
            cols = list(df.columns)
            df["LABEL"] = add_label
            df = df[["LABEL"] + cols]
        return df

    def generate_specs(self, ml_ready=False, rgb=True):
        """Generates spectrograms for each light curve signal in dataframe"""
        if self.df is None:
            self.generate_dataframe(self.files)
        if rgb is True:
            kwargs = self.set_spec_kwargs(cmap="plasma")
        for _, row in self.df.iterrows():
            fname = row["fname"].replace(".fits", "_specgram")
            _, flux = self.read_ts_signal(row["fname"], fmt="kepler.fits", offset=True, remove_nans=True)
            self.flux_specs(
                flux,
                save_for_ml=ml_ready,
                fname=fname,
                title=f"Spectrogram: {row['sc']}-{row['tid']}",
                **kwargs,
            )

    def generate_phase_signal_plots(self):
        """Generates phase-folded light curve plots for each signal in dataframe"""
        if self.df is None:
            self.generate_dataframe(self.files)
        for i in list(range(len(self.df))):
            ts = df.iloc[i]
            self.plot_phase_signals(ts, title_pfx="K2 Phase-folded Light Curve: ", figsize=(11, 5))

    def generate_flux_signal_plots(self):
        """Generates atomic vector plots for each signal in dataframe"""
        if self.df is None:
            self.generate_dataframe(self.files)
        for _, row in self.df.iterrows():
            fname = row["fname"].replace(".fits", "_flux_signal")
            timestamps, flux = self.read_ts_signal(row["fname"], fmt="kepler.fits", offset=True, remove_nans=True)
            self.atomic_vector_plotter(
                flux,
                timestamps=timestamps,
                y_units="PDCSAP Flux (e-/s)",
                x_units="Time (BJD)",
                figsize=(15, 10),
                fname=fname,
                title_pfx=f"K2 Flux Signal: {row['sc']}-{row['tid']}",
            )


# testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path to dataframe (csv file)")
    parser.add_argument("index", type=str, default="index", help="index column name")
    parser.add_argument("-e", "--example", type=str, choices=["svm", "cal"], help="run example demo")
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
