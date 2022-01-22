from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE

class Blueprints:
    def __init__(self, architecture):

        self.input_shape = {
            "memory_classifier": 9,
            "memory_regressor": 9,
            "wallclock_regressor": 9, # X_train.shape[1]
            "ensemble_svm": [9, (3, 128, 128, 3)] # TODO ? check
        }[architecture]

        self.output_shape = {
            "memory_classifier": 4,
            "memory_regressor": 1,
            "wallclock_regressor": 1,
            "ensemble_svm": 1
        }[architecture]
    
        self.layers = {
            "memory_classifier": [18, 32, 64, 32, 18, 9],
            "memory_regressor": [18, 32, 64, 32, 18, 9],
            "wallclock_regressor": [18, 32, 64, 128, 256, 128, 64, 32, 18, 9],
            "ensemble_svm": [18, 32, 64, 32, 18]
        }[architecture]

        self.cost_function = {
            "memory_classifier": "softmax",
            "memory_regressor": "linear",
            "wallclock_regressor": "linear",
            "ensemble_svm": "sigmoid"
        }[architecture]
    
        self.optimizer = {
            "memory_classifier": Adam,
            "memory_regressor": Adam,
            "wallclock_regressor": Adam,
            "ensemble_svm": Adam
        }[architecture]

        self.loss = {
            "memory_classifier": "categorical_crossentropy",
            "memory_regressor": "mse",
            "wallclock_regressor": "mse",
            "ensemble_svm": "binary_crossentropy"
        }[architecture]

        self.activation = {
            "memory_classifier": "relu",
            "memory_regressor": "relu",
            "wallclock_regressor": "relu",
            "ensemble_svm": "leaky_relu",
        }[architecture]

        self.metrics = {
            "memory_classifier": ["accuracy"],
            "memory_regressor": [RMSE(name="rmse")],
            "wallclock_regressor": [RMSE(name="rmse")],
            "ensemble_svm": ["accuracy"]
        }[architecture]

        self.epochs = {
            "memory_classifier": 60,
            "memory_regressor": 60,
            "wallclock_regressor": 300,
            "ensemble_svm": 1000
        }[architecture]

        self.batch_size = {
            "memory_classifier": 32,
            "memory_regressor": 32,
            "wallclock_regressor": 64,
            "ensemble_svm": 32
        }[architecture]

        self.lr_sched = {
            "memory_classifier": False,
            "memory_regressor": False,
            "wallclock_regressor": False,
            "ensemble_svm": True
        }[architecture]
    
        self.input_name = {
            "memory_classifier": "hst_jobs",
            "memory_regressor": "hst_jobs",
            "wallclock_regressor": "hst_jobs",
            "ensemble_svm": "svm_mixed_inputs"
        }[architecture]
    
        self.output_name = {
            "memory_classifier": "memory_classifier",
            "memory_regressor": "memory_regressor",
            "wallclock_regressor": "wallclock_regressor",
            "ensemble_svm": "svm_output"
        }[architecture]

        self.name = {
            "memory_classifier": "mem_clf",
            "memory_regressor": "mem_reg",
            "wallclock_regressor": "wall_reg",
            "ensemble_svm": "ensembleSVM"
        }[architecture]

        self.algorithm = {
            "memory_classifier": "multiclass",
            "memory_regressor": "linreg",
            "wallclock_regressor": "linreg",
            "ensemble_svm": "binary"
        }[architecture]
