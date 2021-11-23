# STANDARD libraries
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import subplots
import plotly.offline as pyo
import plotly.figure_factory as ff
from keras.preprocessing import image
from spacekit.preprocessor.transform import apply_power_transform

plt.style.use("seaborn-bright")
font_dict = {"family": '"Titillium Web", monospace', "size": 16}
mpl.rc("font", **font_dict)


class ImagePlots:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_prime = None
        self.y_prime = None


class Preview(ImagePlots):
    def __init__(self, X, y, X_prime, y_prime):
        super().init(self, X, y)
        self.X_prime = X_prime
        self.y_prime = y_prime

    def preview_augmented(self):
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


class ImagePlots:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_prime = None
        self.y_prime = None


class DataPlots:
    def __init__(self, df, width=1300, height=700, show=True, save_html="."):
        self.df = df
        self.width = width
        self.height = height
        self.show = show
        self.save_html = save_html
        self.target = None  # target (y) column e.g. "label", "memory", "wallclock"
        self.labels = None  #
        self.classes = None  # target classes e.g. [0,1] or [0,1,2,3]
        self.n_classes = None
        self.group = None  # e.g. "detector" or "instr"
        self.categories = None
        self.telescope = None
        self.figures = None
        self.scatter = None
        self.bar = None
        self.groupedbar = None
        self.kde = None

    def feature_subset(self):
        self.categories = {}
        feature_groups = self.df.groupby(self.group)
        for i in list(range(len(feature_groups))):
            dx = feature_groups.get_group(i)
            k = self.gkeys[i]
            self.categories[k] = dx
        return self.categories

    def feature_stats_by_target(self, feature):
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
        show=True,
        save_html=None,
    ):
        if categories is None:
            categories = {"all": self.df}

        scatter_figs = []
        for key, data in categories.items():
            target_groups = data.groupby(self.target)
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
            if show:
                fig.show()
            if save_html:
                if not os.path.exists(self.save_html):
                    os.makedirs(self.save_html, exist_ok=True)
                pyo.plot(
                    fig,
                    filename=f"{save_html}/{key}-{xaxis_name}-{yaxis_name}-scatter.html",
                )
            scatter_figs.append(fig)
        return scatter_figs

    def bar_plots(
        self,
        X,
        Y,
        feature,
        y_err=[None, None],
        width=700,
        height=500,
        cmap=["dodgerblue", "fuchsia"],
        show=True,
        save_html=None,
    ):

        traces = []
        for i in self.classes:
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
        if save_html:
            pyo.plot(fig, filename=f"{save_html}/{feature}-barplot.html")
        if show:
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
        show=True,
        save_html=".",
    ):
        if norm is True:
            df, _ = apply_power_transform(self.df)
            cols = [c + "_scl" for c in cols]
        else:
            df = self.df
        if targets is True:
            hist_data = [df.loc[df[self.target] == c][cols[0]] for c in self.classes]
            group_labels = [f"{cols[0]}={i}" for i in self.labels]
            title = f"KDE {cols[0]} by target class ({self.target})"
            name = f"kde-targets-{cols[0]}.html"
        else:
            hist_data = [df[c] for c in cols]
            group_labels = cols
            title = f"KDE {group_labels[0]} vs {group_labels[1]}"
            name = f"kde-{group_labels[0]}-{group_labels[1]}.html"

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
        if save_html:
            if not os.path.exists(save_html):
                os.makedirs(save_html, exist_ok=True)
            pyo.plot(fig, filename=f"{save_html}/{name}")
        if show:
            fig.show()
        return fig


class SingleVisitMosaic(DataPlots):
    def __init__(self, df, group="det"):
        super().__init__(df)
        self.group = group
        self.telescope = "HST"
        self.target = "label"
        self.classes = list(set(df[self.target].values))  # [0, 1]
        self.labels = ["aligned", "misaligned"]
        self.n_classes = len(set(self.labels))
        self.gkeys = self.group_keys()
        self.categories = self.feature_subset()
        self.continuous = ["rms_ra", "rms_dec", "gaia", "nmatches", "numexp"]
        self.bar = None
        self.scatter = None
        self.kde = None

    def draw_plots(self):
        self.bar = self.alignment_bars()
        self.scatter = self.alignment_scatters()
        self.kde = self.alignment_kde()
        return self

    def alignment_bars(self):
        bars = []
        X = sorted(list(self.gkeys.keys()))
        for f in self.continuous:
            means, errs = self.feature_stats_by_target(f)
            bar = self.bar_plots(X, means, f, y_err=errs, save_html=self.save_html)
            bars.append(bar)
        return bars

    def alignment_scatters(self):
        rms_scatter = self.make_scatter_figs(
            "rms_ra", "rms_dec", categories=self.categories, save_html=self.save_html
        )
        source_scatter = self.make_scatter_figs(
            "point", "segment", categories=self.categories, save_html=self.save_html
        )
        scatters = [rms_scatter, source_scatter]
        return scatters

    def alignment_kde(self):
        cols = self.continuous
        kde_rms = self.kde_plots(["rms_ra", "rms_dec"], save_html=self.save_html)
        kde_targ = [
            self.kde_plots([c], targets=True, save_html=self.save_html) for c in cols
        ]
        kde_norm = [
            self.kde_plots([c], norm=True, targets=True, save_html=self.save_html)
            for c in cols
        ]
        kdes = [kde_rms, kde_targ, kde_norm]
        return kdes

    def group_keys(self):
        if self.group in ["det", "detector"]:
            keys = ["hrc", "ir", "sbc", "uvis", "wfc"]
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
        group_keys = dict(enumerate(keys))
        return group_keys

    # TODO generalize and move up to main class
    def grouped_barplot(self, save=False):
        df = self.df
        groups = df.groupby(["dete_cat"])["label"]
        hrc = groups.get_group(0.0).value_counts()
        ir = groups.get_group(1.0).value_counts()
        sbc = groups.get_group(2.0).value_counts()
        uvis = groups.get_group(3.0).value_counts()
        wfc = groups.get_group(4.0).value_counts()
        trace1 = go.Bar(x=hrc.index, y=hrc, name="HRC", marker=dict(color="red"))
        trace2 = go.Bar(x=ir.index, y=ir, name="IR", marker=dict(color="orange"))
        trace3 = go.Bar(x=sbc.index, y=sbc, name="SBC", marker=dict(color="yellow"))
        trace4 = go.Bar(x=uvis.index, y=uvis, name="UVIS", marker=dict(color="purple"))
        trace5 = go.Bar(x=wfc.index, y=wfc, name="WFC", marker=dict(color="blue"))
        data = [trace1, trace2, trace3, trace4, trace5]
        layout = go.Layout(title="SVM Alignment Labels by Detector")
        fig = go.Figure(data=data, layout=layout)
        if save is True:
            pyo.plot(fig, filename="bar2.html")
        return fig


# TODO
class CalcloudRepro(DataPlots):
    def __init__(self, df, group="instr"):
        super().__init__(df)
        self.telescope = "HST"
        self.target = "mem_bin"
        self.classes = [0, 1, 2, 3]
        self.group = group
        self.labels = ["2g", "8g", "16g", "64g"]
        self.gkeys = self.group_keys()
        self.categories = self.feature_subset()

    def group_keys(self):
        if self.group in ["instr", "instrument"]:
            keys = ["acs", "cos", "stis", "wfc3"]
        elif self.group in ["det", "detector"]:
            keys = ["wfc-uvis", "other"]
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
        svm = SingleVisitMosaic(df)
    else:
        print("More examples coming soon!")
