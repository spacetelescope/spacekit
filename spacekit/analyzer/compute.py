import os
import glob
import pickle
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.python.ops.numpy_ops import np_config

plt.style.use("seaborn-bright")
font_dict = {"family": '"Titillium Web", monospace', "size": 16}
mpl.rc("font", **font_dict)


class Computer(object):
    def __init__(self, algorithm, res_path=None, show=False, validation=False):
        self.algorithm = algorithm  # test/val; clf/reg
        self.res_path = res_path
        self.show = show
        self.validation = validation
        self.model_name = None
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_idx = None
        self.y_pred = None
        self.y_scores = None
        self.y_onehot = None
        self.fnfp = None
        self.cmx = None
        self.cmx_norm = None
        self.cm_fig = None
        self.report = None
        self.roc_auc = None
        self.acc_loss = None
        self.acc_fig = None
        self.loss_fig = None
        self.roc_fig = None
        self.pr_fig = None

    def inputs(self, model, history, X_train, y_train, X_test, y_test, test_idx):
        """Instantiates training vars as attributes. By default, a Computer object is instantiated without these - they are only needed for calculating and storing
        results which can then be retrieved by Computer separately (without training vars) from pickle objects using
        the `upload()` method.

        Parameters
        ----------
        model : object
            Keras functional model
        history : dict
            model training history
        X_train : Pandas dataframe or Numpy array
            training feature data
        y_train : Pandas dataframe or Numpy array
            training target data
        X_test : Pandas dataframe or Numpy array
            test/validation feature data
        y_test : Pandas dataframe or Numpy array
            test/validation target data
        test_idx : Pandas series
            test data index and ground truth values (y_test)

        Returns
        -------
        Computer object (self)
            updated with model attributes used for calculating results
        """
        self.model = model
        self.history = history.history
        self.test_idx = test_idx
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        if self.model_name is None:
            self.model_name = self.model.name
        return self

    def builder_inputs(self, builder=None):
        """produces same result as `inputs` method, using a builder object's attributes instead. Allows for automatic switch to validation set."""
        if builder:
            if self.validation is True:
                try:
                    self.X_test = builder.X_val
                    self.y_test = builder.y_val
                    self.X_train = builder.X_test
                    self.y_train = builder.y_test
                except Exception as e:
                    print(e)
                    print("Validation attributes not set.")
            else:
                self.X_train = builder.X_train
                self.y_train = builder.y_train
                self.X_test = builder.X_test
                self.y_test = builder.y_test
            self.model = builder.model
            self.history = builder.history.history
            self.test_idx = builder.test_idx
            if self.model_name is None:
                self.model_name = builder.model.name
        return self

    def download(self, outputs):
        """Downloads model training results (`outputs` calculated by Computer obj) to local pickle objects for later retrieval and plotting/analysis.

        Args:
            outputs (dict): Outputs created by their respective subclasses (due to distinct diffs btw clf and reg models).
        """
        if self.res_path is None:
            timestamp = int(dt.datetime.now().timestamp())
            datestamp = dt.date.fromtimestamp(timestamp).isoformat()
            prefix = str(datestamp) + "-" + str(timestamp)
            self.res_path = os.path.join(os.getcwd(), prefix, "results", self.algorithm)
        os.makedirs(f"{self.res_path}", exist_ok=True)
        for k, v in outputs.items():
            key = f"{self.res_path}/{k}"
            with open(key, "wb") as pyfi:
                pickle.dump(v, pyfi)
        print(f"Results saved to: {self.res_path}")

    def upload(self):
        """Imports model training results (`outputs` previously calculated by Computer obj) from local pickle objects. These can then be used for plotting/analysis."""
        if self.res_path is None:
            try:
                self.res_path = glob.glob(f"data/*/results/{self.algorithm}")[0]
            except Exception as e:
                print(f"No results found @ {self.res_path} \n", e)
        if not os.path.exists(self.res_path):
            print(f"Path DNE @ {self.res_path}")
        else:
            outputs = {}
            for r in glob.glob(f"{self.res_path}/*"):
                key = r.split("/")[-1]
                with open(r, "rb") as pyfi:
                    outputs[key] = pickle.load(pyfi)
        return outputs

    """ MODEL PERFORMANCE METRICS """

    def onehot_y(self, prefix="lab"):
        """Generates onehot-encoded dataframe of categorical target class values (for multiclassification models).

        Args:
            prefix (str, optional): abbreviated string prefix for target class name. Defaults to "lab" (abbr for "label").

        Returns:
            Pandas Dataframe: Dummy-coded data
        """
        self.y_onehot = pd.get_dummies(self.y_test.ravel(), prefix=prefix)
        return self.y_onehot

    def score_y(self):
        """Probability scores for classification model predictions (`y_pred` probabilities)

        Returns:
            numpy.ndarray: y_scores probabilities array
        """
        self.y_scores = self.model.predict(self.X_test)
        if self.y_scores.shape[1] < 2:
            self.y_scores = np.concatenate(
                [np.round(1 - self.y_scores), np.round(self.y_scores)], axis=1
            )
        return self.y_scores

    def acc_loss_scores(self):
        """Calculate overall accuracy and loss metrics of training and test sets.

        Returns:
            Dictionary: mean accuracy and loss scores of training and test sets (generated via Keras history)
        """
        train_scores = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        test_scores = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        train_acc = np.round(train_scores[1], 2)
        train_loss = np.round(train_scores[0], 2)
        test_acc = np.round(test_scores[1], 2)
        test_loss = np.round(test_scores[0], 2)
        self.acc_loss = {
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        return self.acc_loss

    """ PLOTS """

    def draw_plots(self):
        """Generate standard classification model plots (keras accuracy and loss, ROC-AUC curve, Precision-Recall curve, Confusion Matrix)

        Returns
        -------
        Computer object
            updated with standard plot attribute values
        """

        self.acc_fig = self.keras_acc_plot()
        self.loss_fig = self.keras_loss_plot()
        self.roc_fig = self.make_roc_curve()
        self.pr_fig = self.make_pr_curve()
        self.cm_fig, self.cmx_norm = self.fusion_matrix(self.cmx, self.classes)
        return self

    # Matplotlib "static" alternative to interactive plotly version
    def roc_plots(self):
        """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC).

        Returns
        -------
        int
            roc_auc_score (via sklearn)
        Figure
            receiver-operator characteristic area under the curve (ROC-AUC) plot
        """
        y_true = self.y_test.flatten()
        y_hat = self.model.predict(self.X_test)

        fpr, tpr, thresholds = roc_curve(y_true, y_hat)

        # Threshold Cutoff for predictions
        crossover_index = np.min(np.where(1.0 - fpr <= tpr))
        crossover_cutoff = thresholds[crossover_index]
        crossover_specificity = 1.0 - fpr[crossover_index]
        roc = roc_auc_score(y_true, y_hat)
        print(f"ROC AUC SCORE: {roc}")

        fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(thresholds, 1.0 - fpr)
        ax.plot(thresholds, tpr)
        ax.set_title(
            "Crossover at {0:.2f}, Specificity {1:.2f}".format(
                crossover_cutoff, crossover_specificity
            )
        )

        ax = axes[1]
        ax.plot(fpr, tpr)
        ax.set_title(
            "ROC area under curve: {0:.2f}".format(roc_auc_score(y_true, y_hat))
        )
        if self.show:
            fig.show()

        return roc, fig

    def make_roc_curve(self):

        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{self.y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        fig.update_layout(
            title_text="ROC-AUC",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        if self.show:
            fig.show()
        return fig

    def make_pr_curve(self):

        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=1, y1=0)

        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            auc_score = average_precision_score(y_true, y_score)

            name = f"{self.y_onehot.columns[i]} (AP={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))

        fig.update_layout(
            title_text="Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        if self.show:
            fig.show()
        return fig

    def keras_acc_plot(self):
        keys = list(self.history.keys())
        acc_train = self.history[keys[0]]
        acc_test = self.history[keys[2]]
        n_epochs = list(range(len(acc_train)))
        data = [
            go.Scatter(
                x=n_epochs,
                y=acc_train,
                name=f"Training {keys[0].title()}",
                marker=dict(color="#119dff"),
            ),
            go.Scatter(
                x=n_epochs,
                y=acc_test,
                name=f"Test {keys[0].title()}",
                marker=dict(color="#66c2a5"),
            ),
        ]
        layout = go.Layout(
            title=f"{keys[0].title()}",
            xaxis={"title": "n_epochs"},
            yaxis={"title": "score"},
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        if self.show:
            fig.show()
        return fig

    def keras_loss_plot(self):
        keys = list(self.history.keys())
        loss_train = self.history[keys[1]]
        loss_test = self.history[keys[3]]
        n_epochs = list(range(len(loss_train)))

        data = [
            go.Scatter(
                x=n_epochs,
                y=loss_train,
                name=f"Training {keys[0].title()}",
                marker=dict(color="#119dff"),
            ),
            go.Scatter(
                x=n_epochs,
                y=loss_test,
                name=f"Test {keys[0].title()}",
                marker=dict(color="#66c2a5"),
            ),
        ]
        layout = go.Layout(
            title="Loss",
            xaxis={"title": "n_epochs"},
            yaxis={"title": "score"},
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        if self.show:
            fig.show()
        return fig

    def resid_plot(self):
        """Plot the residual error for a regression model.

        Returns:
            Plotly figure object: interactive plotly scatter fig
        """
        if self.predictions is not None:
            y = self.predictions[:, 1]
            p = self.predictions[:, 0]
        else:
            np_config.enable_numpy_behavior()
            y = self.y_test.reshape(1, -1)
            p = self.y_pred

        data = go.Scatter(x=y, y=p, name="y-y_hat", marker=dict(color="red"))
        layout = go.Layout(
            title="Residual Error",
            xaxis={"title": "y (ground truth)"},
            yaxis={"title": "y_hat (prediction)"},
            width=800,
            height=500,
            paper_bgcolor="#242a44",
            plot_bgcolor="#242a44",
            font={"color": "#ffffff"},
        )
        fig = go.Figure(data=data, layout=layout)
        fig.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=y.min(),
            y0=y.min(),
            x1=y.max(),
            y1=y.max(),
        )
        if self.show is True:
            fig.show()
        return fig

    # Matplotlib "static" alternative to interactive plotly version
    def fusion_matrix(self, cm, classes, normalize=True, cmap="Blues"):
        """
        Confusion Matrix. Can pass in matrix or a tuple (ytrue,ypred) to create on the fly
        classes: class names for target variables
        """
        # make matrix if tuple passed to matrix:
        if isinstance(cm, tuple):
            y_true = cm[0].copy()
            y_pred = cm[1].copy()

            if y_true.ndim > 1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim > 1:
                y_pred = y_pred.argmax(axis=1)
            fusion = confusion_matrix(y_true, y_pred)
        else:
            fusion = cm
        # INTEGER LABELS
        if classes is None:
            classes = list(range(len(fusion)))

        if normalize:
            fusion = fusion.astype("float") / fusion.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        # PLOT
        fig, _ = plt.subplots(figsize=(10, 10))
        plt.imshow(fusion, cmap=cmap, aspect="equal")

        # Add title and axis labels
        plt.title("Confusion Matrix")
        plt.ylabel("TRUE")
        plt.xlabel("PRED")

        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Text formatting
        fmt = ".2f" if normalize else "d"
        # Add labels to each cell
        thresh = fusion.max() / 2.0
        # iterate thru matrix and append labels
        for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
            plt.text(
                j,
                i,
                format(fusion[i, j], fmt),
                horizontalalignment="center",
                color="white" if fusion[i, j] > thresh else "black",
                size=14,
                weight="bold",
            )

        # Add a legend
        plt.colorbar()
        if self.show:
            fig.show()

        return fig, fusion


class ComputeClassifier(Computer):
    def __init__(
        self,
        algorithm="clf",
        classes=["2g", "8g", "16g", "64g"],
        res_path="results/mem_bin",
        show=False,
        validation=False,
    ):
        super().__init__(algorithm, res_path=res_path, show=show, validation=validation)
        self.classes = classes

    def make_outputs(self, dl=True):
        outputs = {
            "y_onehot": self.y_onehot,
            "y_scores": self.y_scores,
            "y_pred": self.y_pred,
            "cmx": self.cmx,
            "cmx_norm": self.cmx_norm,
            "fnfp": self.fnfp,
            "test_idx": self.test_idx,
            "roc_auc": self.roc_auc,
            "acc_loss": self.acc_loss,
            "report": self.report,
        }
        if self.validation is False:
            outputs["history"] = self.history
        if dl:
            super().download(outputs)
        return outputs

    def load_results(self, outputs):
        self.y_onehot = outputs["y_onehot"]
        self.y_scores = outputs["y_scores"]
        self.y_pred = outputs["y_pred"]
        self.cmx = outputs["cmx"]
        self.cmx_norm = outputs["cmx_norm"]
        self.report = outputs["report"]
        self.roc_auc = outputs["roc_auc"]
        self.acc_loss = outputs["acc_loss"]
        if "fnfp" in outputs:
            self.fnfp = outputs["fnfp"]
        self.roc_fig = self.make_roc_curve()
        self.pr_fig = self.make_pr_curve()
        self.cm_fig, _ = self.fusion_matrix(self.cmx, self.classes)
        if self.validation is False:
            self.history = outputs["history"]
            self.acc_fig = self.keras_acc_plot()
            self.loss_fig = self.keras_loss_plot()
        return self

    def track_fnfp(self):
        if self.test_idx is None:
            print("Test index not found")
            return
        try:
            conf_idx = np.where(self.y_pred != self.test_idx.values)
        except AttributeError as e:
            print(
                f"Test/Val Index should be a pandas series, not {type(self.test_idx)}"
            )
            print(e)
            return
        pred_proba = np.asarray(self.model.predict(self.X_test).flatten(), "float32")
        conf_proba = pred_proba[conf_idx]
        fn_idx = self.test_idx.iloc[conf_idx].loc[self.test_idx == 1].index
        fp_idx = self.test_idx.iloc[conf_idx].loc[self.test_idx == 0].index
        self.fnfp = {
            "pred_proba": pred_proba,
            "conf_idx": conf_idx,
            "conf_proba": conf_proba,
            "fn_idx": fn_idx,
            "fp_idx": fp_idx,
        }
        return self.fnfp

    def print_summary(self):
        print(f"\n CLASSIFICATION REPORT: \n{self.report}")
        print(f"\n ACC/LOSS: {self.acc_loss}")
        print(f"\n ROC_AUC: {self.roc_auc}")
        print(f"\nFalse -/+\n{self.cmx}")
        print(f"\nFalse Negatives Index\n{self.fnfp['fn_idx']}\n")
        print(f"\nFalse Positives Index\n{self.fnfp['fp_idx']}\n")


class ComputeBinary(ComputeClassifier):
    def __init__(
        self,
        builder=None,
        algorithm="clf",
        classes=["aligned", "misaligned"],
        res_path="results/svm",
        show=False,
        validation=False,
    ):
        super().__init__(
            algorithm=algorithm,
            classes=classes,
            res_path=res_path,
            show=show,
            validation=validation,
        )
        self.builder_inputs(builder=builder)

    def calculate_results(self, show_summary=True):
        self.y_onehot = self.onehot_y()
        self.y_scores = self.score_y()
        self.y_pred = self.y_scores[:, 1]
        self.report = classification_report(
            self.y_test,
            self.y_pred,
            labels=list(range(len(self.classes))),
            target_names=self.classes,
        )
        self.roc_auc = roc_auc_score(self.y_test, self.y_pred)
        self.acc_loss = self.acc_loss_scores()
        self.cmx = confusion_matrix(self.y_test, self.y_pred)
        self.cmx_norm = self.fusion_matrix(self.cmx, self.classes)[1]
        self.fnfp = self.track_fnfp()
        if show_summary:
            self.print_summary()
        return self


class ComputeMulti(ComputeClassifier):
    def __init__(
        self,
        builder=None,
        algorithm="clf",
        classes=["2g", "8g", "16g", "64g"],
        res_path="results/mem_bin",
        show=False,
        validation=False,
    ):
        super().__init__(
            algorithm=algorithm,
            classes=classes,
            res_path=res_path,
            show=show,
            validation=validation,
        )
        if builder:
            self.inputs(
                builder.model,
                builder.history,
                builder.X_train,
                builder.y_train,
                builder.X_test,
                builder.y_test,
                builder.test_idx,
            )

    def calculate_multi(self, show_summary=True):
        self.y_onehot = self.onehot_multi()
        self.y_scores = self.model.predict(self.X_test)
        self.y_pred = np.round(self.y_scores)
        self.report = classification_report(
            self.y_test,
            self.y_pred,
            labels=list(range(len(self.classes))),
            target_names=self.classes,
            zero_division=0,
        )
        self.roc_auc = self.roc_auc_multi()
        self.acc_loss = self.acc_loss_scores()
        self.cmx = confusion_matrix(
            np.argmax(self.y_test, axis=-1), np.argmax(self.y_pred, axis=-1)
        )
        self.cmx_norm = self.fusion_matrix(self.cmx, self.classes)[1]
        self.fnfp = self.fnfp_multi()
        if show_summary:
            self.print_summary()
        return self

    def roc_auc_multi(self):
        self.roc_auc = []
        for i in range(self.y_scores.shape[1]):
            y_true = self.y_onehot.iloc[:, i]
            y_score = self.y_scores[:, i]
            self.roc_auc.append(roc_auc_score(y_true, y_score))
        return self.roc_auc

    def onehot_multi(self, prefix="bin"):
        self.y_onehot = pd.get_dummies(np.argmax(self.y_test, axis=-1), prefix=prefix)
        self.y_onehot.set_index(self.test_idx.index)
        return self.y_onehot

    def fnfp_multi(self):
        if self.test_idx is None:
            print("Test index not found")
            return
        preds = np.argmax(self.y_pred, axis=-1)
        actual = self.test_idx.values.ravel()
        try:
            conf_idx = np.where(preds != actual)[0]
        except AttributeError as e:  # can probably remove this
            print(
                f"Test/Val Index should be a pandas series, not {type(self.test_idx)}"
            )
            print(e)
            return
        pred_proba = np.amax(self.y_scores, axis=-1)
        conf_proba = pred_proba[conf_idx]

        ipsts = pd.DataFrame(list(self.test_idx.index), columns=["ipsts"])
        y_true = pd.DataFrame(actual, columns=["y_true"])
        y_pred = pd.DataFrame(preds, columns=["y_pred"])
        y_proba = pd.DataFrame(pred_proba, columns=["proba"])
        df_proba = pd.concat([y_true, y_pred, y_proba, ipsts], axis=1)
        df_proba = df_proba.iloc[conf_idx]
        # conf_proba = df_proba.loc[conf_idx][['proba', 'ipsts']].to_dict('split')
        fn, fp = {}, {}
        for label in list(range(len(self.classes))):
            idx = df_proba.loc[df_proba["y_true"] == label]
            false_neg = idx.loc[df_proba["y_pred"] < label]["ipsts"]
            if len(false_neg) > 0:
                fn[str(label)] = false_neg
            false_pos = idx.loc[df_proba["y_pred"] > label]["ipsts"]
            if len(false_pos) > 0:
                fp[str(label)] = false_pos
        df_proba.set_index("ipsts", inplace=True, drop=True)
        self.fnfp = {
            "pred_proba": df_proba.to_dict("split"),
            "conf_idx": conf_idx,
            "conf_proba": conf_proba,
            "fn_idx": fn,
            "fp_idx": fp,
        }
        return self.fnfp


class ComputeRegressor(Computer):
    def __init__(
        self,
        builder=None,
        algorithm="reg",
        res_path="results/memory",
        show=False,
        validation=False,
    ):
        super().__init__(
            algorithm=algorithm, res_path=res_path, show=show, validation=validation
        )
        if builder:
            self.inputs(
                builder.model,
                builder.history,
                builder.X_train,
                builder.y_train,
                builder.X_test,
                builder.y_test,
                builder.test_idx,
            )
        self.y_pred = None
        self.predictions = None
        self.residuals = None
        self.loss = None

    def calculate_results(self):
        """Main calling function to compute regression model scores, including residuals, root mean squared error and L2 cost function. Uses parent class method to save and/or load results to/from disk. Once calculated or loaded, other parent class methods can be used to generate various plots.

        Returns:
            Compute object (self): ComputeRegressor object with calculated model evaluation metrics attributes.
        """
        if self.X_test is None:
            print("No training data - please instantiate the inputs.")
            return
        self.y_pred = self.compute_preds()
        self.predictions = self.yhat_matrix()
        self.residuals = self.get_resid()
        self.loss = self.compute_scores()
        return self

    def compute_preds(self):
        """Get predictions (`y_pred`) based on regression model test inputs (`X_test`).

        Returns:
            numpy.ndarray: predicted values for y (target)
        """
        if self.X_test is not None:
            self.y_pred = self.model.predict(self.X_test)
            return self.y_pred

    def yhat_matrix(self):
        """Compare ground-truth and prediction values of a regression model side-by-side. Used for calculating residuals (see `get_resid` method below).

        Returns:
            2d-array: Concatenation of ground truth (`y_test`) and prediction (`y_pred`) arrays.
        """
        if self.y_pred is not None:
            np_config.enable_numpy_behavior()
            np.set_printoptions(precision=2)
            self.predictions = np.concatenate(
                (
                    self.y_pred.reshape(len(self.y_pred), 1),
                    self.y_test.reshape(len(self.y_test), 1),
                ),
                1,
            )
            return self.predictions

    def get_resid(self):
        """Calculate residual error between ground truth (`y_test`) and prediction values of a regression model.
        Residuals are a measure of how far from the regression line the data points are.

        Returns:
            List: residual error values for a given test set
        """
        if self.predictions is not None:
            self.residuals = []
            for p, a in self.predictions:
                # predicted - actual
                r = p - a
                self.residuals.append(r)
            return self.residuals

    def calculate_L2(self, subset=None):
        """Calculate the L2 Normalization score of a regression model.
        L2 norm is the square root of the sum of the squared vector values (also known as the Euclidean norm or Euclidean distance from the origin).
        This metric is often used when fitting ML algorithms as a regularization method to keep the coefficients of the model small, i.e. to make the model less complex.

        Returns:
            Int: L2 norm
        """
        if subset is not None:
            return np.linalg.norm(np.asarray(subset))
        else:
            return np.linalg.norm(self.residuals)

    def compute_scores(self, error_stats=True):
        """Calculate overall loss metrics of training and test sets. Default for regression is MSE (mean squared error) and RMSE (root MSE).
        RMSE is a measure of how spread out the residuals are (i.e. how concentrated the data is around the line of best fit). Note: RMSE is better in terms of reflecting performance when dealing with large error values (penalizes large errors) while MSE tends to be biased for high values.

        Args:
            error_stats (bool, optional): include RMSE and L2 norm for positive and negative groups of residuals in the test set (here "positive" means above the regression line (>0), "negative" means below (<0)). This can be useful when consequences might be more severe for underestimating vs. overestimating.

        Returns:
            Dictionary: model training loss scores (MSE and RMSE)
        """
        if self.X_test is None:
            return None
        train_scores = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        test_scores = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.loss = {
            "train_loss": np.round(train_scores[0], 2),
            "train_rmse": np.round(train_scores[1], 2),
            "test_loss": np.round(test_scores[0], 2),
            "test_rmse": np.round(test_scores[1], 2),
        }
        if error_stats is True and self.residuals is not None:
            pos, neg = [], []
            for r in self.residuals:
                if r > 0:
                    pos.append(r)
                else:
                    neg.append(r)
            self.loss["rmse_pos"] = np.sqrt(np.mean(np.asarray(pos) ** 2))
            self.loss["rmse_neg"] = np.sqrt(np.mean(np.asarray(neg) ** 2))
            self.loss["l2_norm"] = self.calculate_L2()
            self.loss["l2_pos"] = self.calculate_L2(subset=pos)
            self.loss["l2_neg"] = self.calculate_L2(subset=neg)
        return self.loss

    # def resid_plot(self):
    #     """Plot the residual error for a regression model.

    #     Returns:
    #         Plotly figure object: interactive plotly scatter fig
    #     """
    #     if self.predictions is not None:
    #         y = self.predictions[:,1]
    #         p = self.predictions[:,0]
    #     else:
    #         np_config.enable_numpy_behavior()
    #         y = self.y_test.reshape(1,-1)
    #         p = self.y_pred
    #     fig = px.scatter(x=y, y=p, labels={'x': 'ground truth', 'y': 'prediction'})
    #     fig.add_shape(
    #         type="line", line=dict(dash='dash'),
    #         x0=self.y_test.min(), y0=self.y_test.min(),
    #         x1=self.y_test.max(), y1=self.y_test.max()
    #     )
    #     if self.show is True:
    #         fig.show()
    #     return fig

    def make_outputs(self, dl=True):
        """Create a dictionary of results calculated for a regression model. Used for saving results to disk.

        Args:
            dl (bool, optional): download (save) to disk. Defaults to True.

        Returns:
            Dictionary: outputs stored in a single dictionary for convenience.
        """
        outputs = {
            "predictions": self.predictions,
            "test_idx": self.test_idx,
            "residuals": self.residuals,
            "loss": self.loss,
        }
        if self.validation is False:
            outputs["history"] = self.history
        if dl:
            super().download(outputs)
        return outputs

    def load_results(self, outputs):
        """Load previously calculated results/scores into Compute object (for comparing to other models and/or drawing plots).

        Args:
            outputs (dict): dictionary of results (generated via `make_outputs` method above)

        Returns:
            ComputeRegressor object with results attributes
        """
        self.predictions = outputs["predictions"]
        self.loss = outputs["loss"]
        self.residuals = outputs["residuals"]
        self.res_fig = self.resid_plot()
        if "test_idx" in outputs:
            self.test_idx = outputs["test_idx"]
        if self.validation is False:
            self.history = outputs["history"]
            self.acc_fig = self.keras_acc_plot()
            self.loss_fig = self.keras_loss_plot()
            if "kfold" in outputs:
                self.kfold = outputs["kfold"]
        return self
