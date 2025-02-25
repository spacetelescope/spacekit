from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


class Blueprint:
    def __init__(self, architecture):
        self.architecture = architecture
        self.building = self.build_params()
        self.fitting = self.fit_params()

    def build_params(self):
        return {
            "svm_mlp": self.svm_mlp,
            "svm_cnn": self.svm_cnn,
            "hst_mem_clf": self.hst_mem_clf,
            "hst_mem_reg": self.hst_mem_reg,
            "hst_wall_reg": self.hst_wall_reg,
            "jwst_img3_reg": self.jwst_img3_reg,
            "jwst_spec3_reg": self.jwst_spec3_reg,
        }[self.architecture]

    def fit_params(self):
        return {
            "svm_mlp": self.draft_svm_align,
            "svm_cnn": self.draft_svm_align,
            "hst_mem_clf": self.draft_hst_mem_clf,
            "hst_mem_reg": self.draft_hst_mem_reg,
            "hst_wall_reg": self.draft_hst_wall_reg,
            "jwst_img3_reg": self.draft_jwst_img3_reg,
            "jwst_spec3_reg": self.draft_jwst_spec3_reg,
        }[self.architecture]

    def svm_mlp(self):
        return dict(
            input_shape=10,
            output_shape=1,
            layers=[18, 32, 64, 32, 18],
            activation="leaky_relu",
            cost_function="sigmoid",
            lr_sched=True,
            optimizer=Adam,
            loss="binary_crossentropy",
            metrics=["accuracy"],
            input_name="svm_regression_inputs",
            output_name="svm_regression_output",
            name="svm_mlp",
            ensemble=True,
        )

    def svm_cnn(self):
        return dict(
            input_shape=(3, 128, 128, 3),
            output_shape=1,
            layers=[18, 32, 64, 32, 18],
            activation="leaky_relu",
            cost_function="sigmoid",
            lr_sched=True,
            optimizer=Adam,
            loss="binary_crossentropy",
            metrics=["accuracy"],
            input_name="svm_image_inputs",
            output_name="svm_image_output",
            name="svm_cnn",
            ensemble=True,
        )

    def draft_svm_align(self):
        return dict(
            batch_size=32,
            epochs=60,
            lr=1e-4,
            decay=[100000, 0.96],
            early_stopping=None,
            verbose=2,
            ensemble=True,
        )

    def hst_mem_clf(self):
        return dict(
            input_shape=9,
            output_shape=4,
            layers=[18, 32, 64, 32, 18, 9],
            activation="relu",
            cost_function="softmax",
            lr_sched=False,
            optimizer=Adam,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            input_name="hst_jobs",
            output_name="memory_classifier",
            name="mem_clf",
            algorithm="multiclass",
        )

    def draft_hst_mem_clf(self):
        return dict(
            batch_size=32,
            epochs=60,
            lr=1e-4,
            decay=None,
            early_stopping=None,
            verbose=2,
        )

    def hst_mem_reg(self):
        return dict(
            input_shape=9,
            output_shape=1,
            layers=[18, 32, 64, 32, 18, 9],
            activation="relu",
            cost_function="linear",
            lr_sched=False,
            optimizer=Adam,
            loss="mse",
            metrics=[RMSE(name="rmse")],
            input_name="hst_jobs",
            output_name="memory_regressor",
            name="mem_reg",
            algorithm="linreg",
        )

    def draft_hst_mem_reg(self):
        return dict(
            batch_size=32,
            epochs=60,
            lr=1e-4,
            decay=None,
            early_stopping=None,
            verbose=2,
        )

    def hst_wall_reg(self):
        return dict(
            input_shape=9,
            output_shape=1,
            layers=[18, 32, 64, 128, 256, 128, 64, 32, 18, 9],
            activation="relu",
            cost_function="linear",
            lr_sched=False,
            optimizer=Adam,
            loss="mse",
            metrics=[RMSE(name="rmse")],
            input_name="hst_jobs",
            output_name="wallclock_regressor",
            name="wall_reg",
            algorithm="linreg",
        )

    def draft_hst_wall_reg(self):
        return dict(
            batch_size=64,
            epochs=300,
            lr=1e-4,
            decay=None,
            early_stopping=None,
            verbose=2,
        )

    def jwst_img3_reg(self):
        return dict(
            input_shape=18,
            output_shape=1,
            layers=[18, 36, 72, 144, 72, 36, 18, 9],
            activation="relu",
            cost_function="linear",
            lr_sched=True,
            optimizer=Adam,
            loss="mse",
            metrics=[RMSE(name="rmse")],
            input_name="jwst_cal_image3",
            output_name="image3_regressor",
            name="img3_reg",
            algorithm="linreg",
        )

    def draft_jwst_img3_reg(self):
        return dict(
            batch_size=32,
            epochs=200,
            lr=1e-4,
            decay=[100000, 0.96],
            early_stopping='val_rmse',
            verbose=0,
        )

    def jwst_spec3_reg(self):
        return dict(
            input_shape=18,
            output_shape=1,
            layers=[18, 36, 72, 144, 288, 144, 72, 36, 18],
            activation="relu",
            cost_function="linear",
            lr_sched=True,
            optimizer=Adam,
            loss="mse",
            metrics=[RMSE(name="rmse")],
            input_name="jwst_cal_spec3",
            output_name="spec3_regressor",
            name="spec3_reg",
            algorithm="linreg",
        )

    def draft_jwst_spec3_reg(self):
        return dict(
            batch_size=32,
            epochs=200,
            lr=1e-4,
            decay=[100000, 0.96],
            early_stopping='val_rmse',
            verbose=0,
        )
