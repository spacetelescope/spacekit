from tensorflow.keras.optimizers import Adam


class Blueprint:
    def __init__(self, architecture):
        self.architecture = architecture
        self.building = self.build_params()
        self.fitting = self.fit_params()

    def build_params(self):
        return {
            "svm_mlp": self.svm_mlp,
            "svm_cnn": self.svm_cnn,
        }[self.architecture]

    def fit_params(self):
        return {
            "svm_mlp": self.draft_svm_fit,
            "svm_cnn": self.draft_svm_fit,
        }

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

    def draft_svm_fit(self):
        return dict(
            batch_size=32,
            epochs=60,
            lr=1e-4,
            decay=[100000, 0.96],
            early_stopping=None,
            verbose=2,
            ensemble=True,
        )


# from tensorflow.keras.optimizers import Adam, schedules
# from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
# class Blueprint:
#     def __init__(self, architecture):

#         self.input_shape = {
#             "memory_classifier": 9,
#             "memory_regressor": 9,
#             "wallclock_regressor": 9,  # X_train.shape[1]
#             "svm_mlp": 10,
#             "svm_cnn": (3, 128, 128, 3),
#             "svm_ensemble": [(None, 10), (None, 3, 128, 128, 3)]
#         }[architecture]

#         self.output_shape = {
#             "memory_classifier": 4,
#             "memory_regressor": 1,
#             "wallclock_regressor": 1,
#             "svm_mlp": 1,
#             "svm_cnn": 1,
#             "svm_ensemble": 1,
#         }[architecture]

#         self.layers = {
#             "memory_classifier": [18, 32, 64, 32, 18, 9],
#             "memory_regressor": [18, 32, 64, 32, 18, 9],
#             "wallclock_regressor": [18, 32, 64, 128, 256, 128, 64, 32, 18, 9],
#             "svm_mlp": [18, 32, 64, 32, 18],
#             "svm_cnn": [18, 32, 64, 32, 18],
#             "svm_ensemble": [18, 32, 64, 32, 18],
#         }[architecture]

#         self.cost_function = {
#             "memory_classifier": "softmax",
#             "memory_regressor": "linear",
#             "wallclock_regressor": "linear",
#             "svm_mlp": "sigmoid",
#             "svm_cnn": "sigmoid",
#             "svm_ensemble": "sigmoid",
#         }[architecture]

#         self.optimizer = {
#             "memory_classifier": Adam,
#             "memory_regressor": Adam,
#             "wallclock_regressor": Adam,
#             "svm_mlp": Adam,
#             "svm_cnn": Adam,
#             "svm_ensemble": Adam,
#         }[architecture]

#         self.loss = {
#             "memory_classifier": "categorical_crossentropy",
#             "memory_regressor": "mse",
#             "wallclock_regressor": "mse",
#             "svm_mlp": "binary_crossentropy",
#             "svm_cnn": "binary_crossentropy",
#             "svm_ensemble": "binary_crossentropy",
#         }[architecture]

#         self.activation = {
#             "memory_classifier": "relu",
#             "memory_regressor": "relu",
#             "wallclock_regressor": "relu",
#             "svm_mlp": "leaky_relu",
#             "svm_cnn": "leaky_relu",
#             "svm_ensemble": "leaky_relu",
#         }[architecture]

#         self.metrics = {
#             "memory_classifier": ["accuracy"],
#             "memory_regressor": [RMSE(name="rmse")],
#             "wallclock_regressor": [RMSE(name="rmse")],
#             "svm_mlp": ["accuracy"],
#             "svm_cnn": ["accuracy"],
#             "svm_ensemble": ["accuracy"],
#         }[architecture]

#         self.epochs = {
#             "memory_classifier": 60,
#             "memory_regressor": 60,
#             "wallclock_regressor": 300,
#             "svm_mlp": 60,
#             "svm_cnn": 60,
#             "svm_ensemble": 1000,
#         }[architecture]

#         self.batch_size = {
#             "memory_classifier": 32,
#             "memory_regressor": 32,
#             "wallclock_regressor": 64,
#             "svm_mlp": 32,
#             "svm_cnn": 32,
#             "svm_ensemble": 32,
#         }[architecture]

#         self.lr_sched = {
#             "memory_classifier": False,
#             "memory_regressor": False,
#             "wallclock_regressor": False,
#             "svm_mlp": True,
#             "svm_cnn": True,
#             "svm_ensemble": True,
#         }[architecture]

#         self.input_name = {
#             "memory_classifier": "hst_jobs",
#             "memory_regressor": "hst_jobs",
#             "wallclock_regressor": "hst_jobs",
#             "svm_mlp": "svm_regression_inputs",
#             "svm_cnn": "svm_image_inputs",
#             "svm_ensemble": "svm_mixed_inputs",
#         }[architecture]

#         self.output_name = {
#             "memory_classifier": "memory_classifier",
#             "memory_regressor": "memory_regressor",
#             "wallclock_regressor": "wallclock_regressor",
#             "svm_mlp": "svm_regression_output",
#             "svm_cnn": "svm_image_output",
#             "svm_ensemble": "svm_output",
#         }[architecture]

#         self.name = {
#             "memory_classifier": "mem_clf",
#             "memory_regressor": "mem_reg",
#             "wallclock_regressor": "wall_reg",
#             "svm_mlp": "svm_mlp",
#             "svm_cnn": "svm_cnn",
#             "svm_ensemble": "ensembleSVM",
#         }[architecture]

#         self.algorithm = {
#             "memory_classifier": "multiclass",
#             "memory_regressor": "linreg",
#             "wallclock_regressor": "linreg",
#             "svm_mlp": "binary",
#             "svm_cnn": "binary",
#             "svm_ensemble": "binary",
#         }[architecture]

#         self.compile_params = {
#             "memory_classifier": None,
#             "memory_regressor": None,
#             "wallclock_regressor": None,
#             "svm_mlp": None,
#             "svm_cnn": None,
#             "svm_ensemble": dict(
#                 loss="binary_crossentropy",
#                 metrics=["accuracy"],
#                 optimizer=Adam(
#                     learning_rate=schedules.ExponentialDecay(
#                         initial_learning_rate=1e-4,
#                         decay_steps=100000,
#                         decay_rate=0.96,
#                         staircase=True,
#                     )
#                 ),
#             )
#         }[architecture]
