import os
import sys
import pickle
import itertools
import pandas as pd
import numpy as np
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
    jaccard_score,
    accuracy_score,
    recall_score,
    fowlkes_mallows_score,
)

plt.style.use("seaborn-bright")
font_dict = {"family": '"Titillium Web", monospace', "size": 16}
mpl.rc("font", **font_dict)


class Computer:
    def __init__(self, model_name, computation, classes, show=False):
        self.model_name = model_name
        self.computation = computation
        self.classes = classes
        self.show = show
        self.res_path = None
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_idx = None
        self.y_onehot = None
        self.y_scores = None
        self.y_pred = None
        self.cmx = None
        self.fnfp = None
        self.report = None
        self.roc_auc = None
        self.acc_loss = None
        self.acc_fig = None
        self.loss_fig = None
        self.roc_fig = None
        self.pr_fig = None
        self.cm_fig = None
        self.cmx_norm = None

    def inputs(self, model, history, X_train, y_train, X_test, y_test, test_idx):
        self.model = model
        self.history = history.history
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_idx = test_idx
        return self

    def store_results(self):
        # model_name, compute, training_date, etc
        outputs = {
            "predictions": {
                "y_onehot": self.y_onehot,
                "y_scores": self.y_scores,
                "y_pred": self.y_pred,
                "cmx": self.cmx,
                "cmx_norm": self.cmx_norm,
                "fnfp": self.fnfp,
            },
            "scores": {
                "roc_auc": self.roc_auc,
                "acc_loss": self.acc_loss,
                "report": self.report,
            },
        }
        if self.computation == "val":
            outputs["plots"] = {
                "roc_fig": self.roc_fig,
                "pr_fig": self.pr_fig,
                "cm_fig": self.cm_fig,
            }
        else:
            outputs["plots"] = {
                "keras_acc": self.acc_fig,
                "keras_loss": self.loss_fig,
                "roc_fig": self.roc_fig,
                "pr_fig": self.pr_fig,
                "cm_fig": self.cm_fig,
            }
        return outputs

    def download(self):
        outputs = self.store_results()
        if self.res_path is None:
            self.res_path = os.path.join("./results", self.model_name, self.computation)
        os.makedirs(f"{self.res_path}", exist_ok=True)
        for k, v in outputs.items():
            key = f"{self.res_path}/{k}"
            with open(key, "wb") as file_pi:
                pickle.dump(v, file_pi)
        print(f"Results saved to: {self.res_path}")

        out = sys.stdout
        with open(f"{self.res_path}/metrics_report.txt", "w") as f:
            sys.stdout = f
            print(self.report)
            sys.stdout = out

    def upload(self):
        if self.res_path is None:
            self.res_path = os.path.join("./results", self.model_name, self.computation)
        if os.path.exists(self.res_path):
            res = {}
            for r in os.listdir(self.res_path):
                key = os.path.join(self.res_path, r)
                with open(key, "rb") as py_fi:
                    res[r] = pickle.load(py_fi)
            self.outputs(res)
        else:
            print(f"No results found @ {self.res_path}")
        return self

    def outputs(self, res):
        self.y_onehot = res["predictions"]["y_onehot"]
        self.y_scores = res["predictions"]["y_scores"]
        self.y_pred = res["predictions"]["y_pred"]
        self.cmx = res["predictions"]["cmx"]
        self.cmx_norm = res["predictions"]["cmx_norm"]
        self.fnfp = res["predictions"]["fnfp"]
        # self.report = res['scores']['report']
        self.roc_auc = res["scores"]["roc_auc"]
        self.acc_loss = res["scores"]["acc_loss"]
        self.roc_fig = res["plots"]["roc_fig"]
        self.pr_fig = res["plots"]["pr_fig"]
        self.cm_fig = res["plots"]["cm_fig"]
        if self.computation == "test":
            self.acc_fig = res["plots"]["acc_fig"]
            self.loss_fig = res["plots"]["loss_fig"]
        with open(f"{self.res_path}/metrics_report.txt", "r") as f:
            self.report = f.read()
        return self

    """ MODEL PERFORMANCE METRICS """

    def calculate_results(self, show_summary=True):
        self.y_onehot = self.onehot_y()
        self.y_scores = self.score_y()
        self.y_pred = self.y_scores[:, 1]
        self.report = classification_report(
            self.y_test, self.y_pred, labels=[0, 1], target_names=self.classes
        )
        self.roc_auc = roc_auc_score(self.y_test, self.y_pred)
        self.acc_loss = self.acc_loss_scores()
        self.cmx = confusion_matrix(self.y_test, self.y_pred)
        self.fnfp = self.track_fnfp()

        if show_summary:
            self.print_summary()

        return self

    def onehot_y(self):
        self.y_onehot = pd.get_dummies(self.y_test.ravel(), prefix="lab")
        return self.y_onehot

    def score_y(self):
        y_scores = self.model.predict(self.X_test)
        self.y_scores = np.concatenate(
            [np.round(1 - y_scores), np.round(y_scores)], axis=1
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

    """ PLOTS """

    def draw_plots(self):
        self.acc_fig = self.keras_acc_plot()
        self.loss_fig = self.keras_loss_plot()
        self.roc_fig = self.make_roc_curve()
        self.pr_fig = self.make_pr_curve()
        self.cm_fig, self.cmx_norm = self.fusion_matrix(self.cm, self.classes)
        return self

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


class ComputeTest(Computer):
    def __init__(
        self,
        model_name,
        classes,
        model,
        history,
        X_train,
        y_train,
        X_test,
        y_test,
        test_idx,
    ):
        super().__init__(model_name, "test", classes)
        self.inputs(model, history, X_train, y_train, X_test, y_test, test_idx)


class ComputeVal(Computer):
    def __init__(
        self, model_name, classes, model, X_test, y_test, X_val, y_val, val_idx
    ):
        super().__init__(model_name, "val", classes)
        self.inputs(model, X_test, y_test, X_val, y_val, val_idx)


class AnalogComputer:
    """Classic matplotlib plots"""

    def __init__(self, X, y, model, history, verbose=False):
        self.X = X
        self.y = y
        self.model = model
        self.history = history
        self.verbose = False
        self.y_pred = None
        self.fnfp_dict = None
        self.keras_fig = None
        self.cmx = None
        self.roc_fig = None
        self.results = None

    def compute(
        self,
        preds=True,
        summary=True,
        cmx=True,
        classes=None,
        report=True,
        roc=True,
        hist=True,
    ):
        """
        evaluates model predictions and stores the output in a dict
        returns `results`
        """
        res = {}
        res["model"] = self.model.name

        # class predictions
        if preds:
            res["preds"] = self.get_preds()

        if summary:
            res["summary"] = self.model.summary

        # FUSION MATRIX
        if cmx:
            if classes is None:
                classes = set(self.y)
                # classes=['0','1']
            else:
                classes = classes
            # Plot fusion matrix
            res["FM"] = self.fusion_matrix(
                matrix=(self.y.flatten(), self.y_pred), classes=classes
            )

        # ROC Area Under Curve
        if roc:
            res["ROC"] = self.roc_plots(self.X, self.y, self.model)

        # CLASSIFICATION REPORT
        if report:
            num_dashes = 20
            print("\n")
            print("---" * num_dashes)
            print("\tCLASSIFICATION REPORT:")
            print("---" * num_dashes)
            # generate report
            res["report"] = classification_report(self.y.flatten(), self.y_pred)
            print(report)

        # save to dict:
        res["jaccard"] = jaccard_score(self.y, self.y_pred)
        res["fowlkes"] = fowlkes_mallows_score(self.y, self.y_pred)
        res["accuracy"] = accuracy_score(self.y, self.y_pred)
        res["recall"] = recall_score(self.y, self.y_pred)

        # Plot Model Training Results (PLOT KERAS HISTORY)
        if hist:
            res["HIST"] = self.keras_history(self.history)
        return res

    @staticmethod
    def get_preds(self):
        # class predictions
        # self.y_pred = self.model.predict_classes(self.X).flatten()
        self.y_pred = np.round(self.model.predict(self.X))
        if self.verbose:
            pred_count = pd.Series(self.y_pred).value_counts(normalize=False)
            print(f"y_pred:\n {pred_count}")
            print("\n")
        return self.y_pred

    def fnfp(self, training=False):
        if self.y_pred is None:
            self.y_pred = np.round(self.model.predict(self.X))

        pos_idx = self.y == 1
        neg_idx = self.y == 0

        # tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
        fn = np.sum(self.y_pred[pos_idx] == 0) / self.y_pred.shape[0]
        # tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
        fp = np.sum(self.y_pred[neg_idx] == 1) / self.y_pred.shape[0]

        if training:
            print(f"FN Rate (Training): {round(fn*100,4)}%")
            print(f"FP Rate (Training): {round(fp*100,4)}%")
        else:
            print(f"FN Rate (Test): {round(fn*100,4)}%")
            print(f"FP Rate (Test): {round(fp*100,4)}%")

        self.fnfp_dict = {"fn": fn, "fp": fp}
        return self.fnfp_dict

    def keras_history(self, figsize=(15, 6), show=True):
        """
        side by side sublots of training val accuracy and loss (left and right respectively)
        """
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(self.history.history["accuracy"])
        ax.plot(self.history.history["val_accuracy"])
        ax.set_title("Model Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend(["Train", "Test"], loc="upper left")

        ax = axes[1]
        ax.plot(self.history.history["loss"])
        ax.plot(self.history.history["val_loss"])
        ax.set_title("Model Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.legend(["Train", "Test"], loc="upper left")
        if show is True:
            fig.show()
        return fig

    def fusion_matrix(self, matrix, classes=None, normalize=True, cmap="Blues"):
        # make matrix if tuple passed to matrix:
        if isinstance(matrix, tuple):
            y_true = matrix[0].copy()
            y_pred = matrix[1].copy()

            if y_true.ndim > 1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim > 1:
                y_pred = y_pred.argmax(axis=1)
            fusion = confusion_matrix(y_true, y_pred)
        else:
            fusion = matrix

        # INTEGER LABELS
        if classes is None:
            classes = list(range(len(matrix)))

        # NORMALIZING
        # Check if normalize is set to True
        # If so, normalize the raw fusion matrix before visualizing
        if normalize:
            fusion = fusion.astype("float") / fusion.sum(axis=1)[:, np.newaxis]
            thresh = 0.5
            fmt = ".2f"
        else:
            fmt = "d"
            thresh = fusion.max() / 2.0

        # PLOT
        fig = plt.subplots(figsize=(10, 10))
        plt.imshow(fusion, cmap=cmap, aspect="equal")

        # Add title and axis labels
        plt.title("Confusion Matrix")
        plt.ylabel("TRUE")
        plt.xlabel("PRED")

        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        # ax.set_ylim(len(fusion), -.5,.5) ## <-- This was messing up the plots!

        # Text formatting
        fmt = ".2f" if normalize else "d"
        # Add labels to each cell
        # thresh = fusion.max() / 2.
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
        plt.show()
        return fusion, fig

    def roc_plots(self):
        """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC)

        Arguments:
            X {feature set} -- typically X_test
            y {labels} -- typically y_test
            model {classifier} -- the model name for which you are calculting roc score

        Returns:
            roc -- roc_auc_score (via sklearn)
        """
        fpr, tpr, thresholds = roc_curve(self.y.flatten(), self.y_pred)

        # Threshold Cutoff for predictions
        crossover_index = np.min(np.where(1.0 - fpr <= tpr))
        crossover_cutoff = thresholds[crossover_index]
        crossover_specificity = 1.0 - fpr[crossover_index]

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
            "ROC area under curve: {0:.2f}".format(
                roc_auc_score(self.y.flatten(), self.y_pred)
            )
        )
        fig.show()

        roc = roc_auc_score(self.y.flatten(), self.y_pred)

        return roc
