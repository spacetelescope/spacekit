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
import plotly.express as px
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import mean_squared_error as MSE
plt.style.use("seaborn-bright")
font_dict = {"family": '"Titillium Web", monospace', "size": 16}
mpl.rc("font", **font_dict)

from spacekit.preprocessor.transform import tensors_to_arrays


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
        """Instantiates training vars as attributes (typically carried over from a Builder object post-training).
        By default, a Computer object is instantiated without these - they are only needed for calculating and storing
        results which can then be retrieved by Computer separately (without training vars) from pickle objects using
        the `upload()` method.

        Args:
            model (object): Keras functional model
            history (dict): model training history
            X_train (dataframe or array): training feature data
            y_train (dataframe or array): training target data
            X_test (dataframe or array): test feature data
            y_test (dataframe or array): test target data
            test_idx (pandas Series): test data index and ground truth values (used for FNFP)

        Returns:
            class object (self): updated Computer object with model attributes used for calculating results
        """
        self.model = model
        self.history = history.history
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_idx = test_idx
        if self.model_name is None:
            self.model_name = self.model.name
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
        self.y_onehot = pd.get_dummies(self.y_test.ravel(), prefix=prefix)
        return self.y_onehot

    def score_y(self):
        self.y_scores = self.model.predict(self.X_test)
        if self.y_scores.shape[1] < 2:
            self.y_scores = np.concatenate(
                [np.round(1 - self.y_scores), np.round(self.y_scores)], axis=1
            )
        return self.y_scores

    def acc_loss_scores(self):
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
        self.acc_fig = self.keras_acc_plot()
        self.loss_fig = self.keras_loss_plot()
        self.roc_fig = self.make_roc_curve()
        self.pr_fig = self.make_pr_curve()
        self.cm_fig, self.cmx_norm = self.fusion_matrix(self.cmx, self.classes)
        return self

    # Matplotlib "static" alternative to interactive plotly version
    def roc_plots(self):
        """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC)

        Arguments:
            X {feature set} -- typically X_test
            y {labels} -- typically y_test
            model {classifier} -- the model name for which you are calculting roc score

        Returns:
            roc -- roc_auc_score (via sklearn)
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
        acc_train = self.history["accuracy"]
        acc_test = self.history["val_accuracy"]
        n_epochs = list(range(len(acc_train)))
        data = [
            go.Scatter(
                x=n_epochs,
                y=acc_train,
                name="Training Accuracy",
                marker=dict(color="#119dff"),
            ),
            go.Scatter(
                x=n_epochs,
                y=acc_test,
                name="Test Accuracy",
                marker=dict(color="#66c2a5"),
            ),
        ]
        layout = go.Layout(
            title="Accuracy",
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
        loss_train = self.history["loss"]
        loss_test = self.history["val_loss"]
        n_epochs = list(range(len(loss_train)))

        data = [
            go.Scatter(
                x=n_epochs,
                y=loss_train,
                name="Training Loss",
                marker=dict(color="#119dff"),
            ),
            go.Scatter(
                x=n_epochs, y=loss_test, name="Test Loss", marker=dict(color="#66c2a5")
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

    # Matplotlib "static" alternative to interactive plotly version
    def fusion_matrix(self, cm, classes, normalize=True, cmap="Blues"):
        """
        FUSION MATRIX!
        -------------

        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly
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


class ComputeTest(ComputeClassifier):
    def __init__(
        self,
        model,
        history,
        X_train,
        y_train,
        X_test,
        y_test,
        test_idx,
        classes=["aligned", "misaligned"],
        res_path="results/test",
        show=False,
        validation=False,
    ):
        super().__init__(
            algorithm="clf",
            classes=classes,
            res_path=res_path,
            show=show,
            validation=validation,
        )
        self.inputs(model, history, X_train, y_train, X_test, y_test, test_idx)
        self.classes = classes


class ComputeVal(ComputeClassifier):
    def __init__(
        self,
        model,
        X_test,
        y_test,
        X_val,
        y_val,
        val_idx,
        classes=["aligned", "misaligned"],
        res_path="results/val",
        show=False,
        validation=True,
    ):
        super().__init__(
            algorithm="clf",
            classes=classes,
            res_path=res_path,
            show=show,
            validation=validation,
        )
        self.inputs(model, X_test, y_test, X_val, y_val, val_idx)


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
        if builder:
            self.inputs(
                builder.model,
                builder.X_test,
                builder.y_test,
                builder.X_val,
                builder.y_val,
                builder.test_idx,
            )

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
    def __init__(self, builder=None, algorithm="reg", res_path="results/memory", show=False, validation=False):
        super().__init__(algorithm=algorithm, res_path=res_path, show=show, validation=validation)
        if builder:
            self.inputs(
                builder.model,
                builder.history,
                builder.X_train,
                builder.y_train,
                builder.X_test,
                builder.y_test,
                builder.test_idx
            )
        self.y_pred = None
        self.scores = None
        self.predictions = None
        self.residuals = None
        self.rmse = None


    def calculate_results(self):
        """Main calling function to compute regression model scores, including residuals, root mean squared error and L2 cost function. Uses parent class method to save and/or load results to/from disk. Once calculated or loaded, other parent class methods can be used to generate various plots.

        Returns:
            Compute object: ComputeRegressor object for quick evaluation of model performance. 
        """
        if self.X_test is None:
            print("No training data - please instantiate the inputs.")
            return
        self.acc_loss = self.get_scores()
        self.y_pred = self.model.predict(self.X_test)
        self.X_train, self.y_train, self.X_test, self.y_test = tensors_to_arrays(self.X_train, self.y_train, self.X_test, self.y_test)
        self.predictions = self.predict_reg()
        self.residuals = self.get_resid()
        self.rmse = self.calculate_rmse()
        return self

    def get_scores(self):
        """Calculates training and test set accuracy and loss scores. Loss is a more important metric for regression models, but accuracy is included for consistency with other compute objects (and for thoroughness). Use `calculate_rmse` for Root Mean Squared Error calculations.

        Returns:
            Dictionary: model training accuracy and loss scores based on Keras history
        """
        train_scores = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        test_scores = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        train_acc = np.round(train_scores[1], 2)
        train_loss = np.round(train_scores[0], 2)
        test_acc = np.round(test_scores[1], 2)
        test_loss = np.round(test_scores[0], 2)
        self.acc_loss = {"train_acc": train_acc, "train_loss": train_loss, "test_acc": test_acc, "test_loss": test_loss}
        return self.acc_loss

    def predict_reg(self):
        """Compare ground-truth and prediction values of a regression model side-by-side. Used for calculating residuals (see `get_resid` method below).

        Returns:
            2d-array: Concatenation of ground truth (`y_test`) and prediction (`y_pred`) arrays.
        """
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        np.set_printoptions(precision=2)
        self.predictions = np.concatenate((self.y_pred.reshape(len(self.y_pred), 1), self.y_test.reshape(len(self.y_test), 1)), 1)
        return self.predictions

    def get_resid(self):
        """Calculate residual error between ground truth (`y_test`) and prediction values of a regression model. 
        Residuals are a measure of how far from the regression line the data points are.

        Returns:
            Array: residual error values for a given test set
        """
        self.residuals = []
        if self.predictions is None:
            self.predictions = self.predict_reg()
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
        if self.residuals is None:
            self.residuals = self.get_resid()
        self.L2 = np.linalg.norm(self.residuals)
        print("Cost (L2 Norm): ", self.L2)
        return self.L2
    
    def calculate_rmse(self, subsets=True, incl_train=True, L2=True):
        """Compute the Root Mean Squared Error (RMSE) of a regression model (using test set).
        RMSE is a measure of how spread out the residuals are (i.e. how concentrated the data is around the line of best fit).

        Args:
            subsets (bool, optional): also calculate RMSE for subsets of residuals above and below regression line. Defaults to True.
            incl_train (bool, optional): also calculate RMSE for training set. Defaults to True.
            L2 (bool, optional): also calculate L2 Norm

        Returns:
            Dictionary: RMSE for test set (plus training, pos and neg subsets, and L2 norm values if set to `True`).
        """
        # RMSE Computation
        if self.y_test is None:
            if self.predictions:
                self.y_test = self.predictions[:, 1]
                self.y_pred = self.predictions[:, 0]
            elif self.X_test:
                self.y_pred = self.model.predict(self.X_test)
            else:
                print("y_test and y_pred values are required for RMSE.")
                return
        rmse_test = np.sqrt(MSE(self.y_test, self.y_pred))
        print("RMSE Test : % f" % (rmse_test))
        self.rmse = {"rmse_test": rmse_test}
        # include training set rmse if available
        if incl_train is True:
            if self.X_train:
                y_hat = self.model.predict(self.X_train)
                rmse_train = np.sqrt(MSE(self.y_train, y_hat))
                print("RMSE Train : % f" % (rmse_train))
                self.rmse["rmse_train"] = rmse_train
        if subsets is True:
            if self.residuals is None:
                self.residuals = self.get_resid()
            over, under = [], []
            for r in self.residuals:
                if r > 0:
                    over.append(r)
                else:
                    under.append(r)
            self.rmse["rmse_over"] = np.sqrt(np.mean(np.asarray(over)**2))
            # subset below regression line (underestimations)
            self.rmse["rmse_under"] = np.sqrt(np.mean(np.asarray(under)**2))
        if L2 is True:
            self.rmse["L2_norm"] = self.calculate_L2()
            self.rmse["L2_over"] = self.calculate_L2(subset=over)
            self.rmse["L2_under"] = self.calculate_L2(subset=under)
        return self.rmse
    
    def resid_plot(self):
        """Plot the residual error for a regression model.

        Args:
            show (bool, optional): Show the interactive plot. Defaults to False.

        Returns:
            Plotly figure object: interactive plotly scatter fig  
        """
        if self.predictions is not None:
            y = self.predictions[:,1]
            p = self.predictions[:,0]
        else:
            y = self.y_test.reshape(1,-1)
            p = self.y_pred
        fig = px.scatter(x=y, y=p, labels={'x': 'ground truth', 'y': 'prediction'})
        fig.add_shape(
            type="line", line=dict(dash='dash'),
            x0=self.y_test.min(), y0=self.y_test.min(),
            x1=self.y_test.max(), y1=self.y_test.max()
        )
        if self.show is True:
            fig.show()
        return fig

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
            "acc_loss": self.acc_loss,
            "residuals": self.residuals,
            "rmse": self.rmse
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
        self.acc_loss = outputs["acc_loss"]
        self.residuals = outputs["residuals"]
        self.rmse = outputs["rmse"]
        if "test_idx" in outputs:
            self.test_idx = outputs["test_idx"]
        if self.validation is False:
            self.history = outputs["history"]
            self.acc_fig = self.keras_acc_plot()
            self.loss_fig = self.keras_loss_plot()
            if "kfold" in outputs:
                self.kfold = outputs["kfold"]
        return self
