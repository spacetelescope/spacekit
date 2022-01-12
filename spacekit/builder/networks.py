# STANDARD libraries
import os
import importlib.resources
import numpy as np
import time
import datetime as dt
from tensorflow.keras.utils import plot_model  # req: pydot, graphviz
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.layers import (
    Dense,
    Input,
    concatenate,
    Conv1D,
    MaxPool1D,
    Dropout,
    Flatten,
    BatchNormalization,
    Conv3D,
    MaxPool3D,
    GlobalAveragePooling3D,
)
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
from spacekit.generator.augment import augment_data, augment_image
from spacekit.analyzer.track import stopwatch

DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2

# TODO: K-fold cross-val class


class Builder:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.blueprint = "ensemble"  # "ensemble", "cnn3d", "mlp" "cnn2d"
        self.batch_size = 32
        self.epochs = 60
        self.lr = 1e-4
        self.decay = [100000, 0.96]
        self.early_stopping = None
        self.callbacks = None
        self.verbose = 2
        self.ensemble = False
        self.model = None
        self.mlp = None
        self.cnn = None
        self.ann = None
        self.make_batches = None
        self.steps_per_epoch = None
        self.history = None
        self.name = None
        self.model_path = None

    # TODO: add params for loading other default pkg models (hstcal)
    def load_saved_model(self):
        if self.model_path is None:
            model_src = "spacekit.skopes.trained_networks"
            with importlib.resources.path(model_src, "ensembleSVM") as mod:
                self.model_path = mod
        model = load_model(self.model_path)
        self.model = Model(inputs=model.inputs, outputs=model.outputs)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.decay_learning_rate()),
            metrics=["accuracy"],
        )
        return self

    # def unzip_model_files(path_to_zip, extract_to="."):
    #     model_base = os.path.basename(path_to_zip).split(".")[0]
    #     with ZipFile(path_to_zip, "r") as zip_ref:
    #         zip_ref.extractall(extract_to)
    #     model_path = os.path.join(extract_to, model_base)
    #     return model_path

    def build_params(
        self,
        input_shape,
        kernel_size,
        activation,
        strides,
        optimizer,
        lr,
        decay,
        early_stopping,
        loss,
        metrics,
    ):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.metrics = metrics
        self.decay = decay
        self.early_stopping = early_stopping
        return self

    def design(self):
        if self.blueprint == "mlp":
            return dict(
                batches=self.batch_mlp(), steps=self.X_train.shape[0] // self.batch_size
            )
        elif self.blueprint == "image3d":
            return dict(
                batches=self.batch_cnn(), steps=self.X_train.shape[0] // self.batch_size
            )
        elif self.blueprint == "ensemble":
            return dict(
                batches=self.batch_ensemble(),
                steps=self.X_train[0].shape[0] // self.batch_size,
            )
        elif self.blueprint == "cnn2d":
            return dict(
                batches=self.batch_maker(),
                steps=self.X_train.shape[1] // self.batch_size,
            )

    def batch_steps(self):
        design = self.design()
        make_batches = design["batches"]
        steps_per_epoch = design["steps"]
        return make_batches, steps_per_epoch

    def fit_params(
        self,
        batch_size=32,
        epochs=60,
        lr=1e-4,
        decay=[100000, 0.96],
        early_stopping=None,
        verbose=2,
        ensemble=False,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.ensemble = ensemble
        return self

    def decay_learning_rate(self):
        """set learning schedule with exponential decay
        lr (initial learning rate: 1e-4
        decay: [decay_steps, decay_rate]
        """
        lr_schedule = optimizers.schedules.ExponentialDecay(
            self.lr, decay_steps=self.decay[0], decay_rate=self.decay[1], staircase=True
        )
        return lr_schedule

    def set_callbacks(self):
        """
        early_stopping:
            clf: 'val_accuracy' or 'val_loss'
            reg: 'val_loss' or 'val_rmse'
        """
        model_name = str(self.model.name_scope().rstrip("/").upper())
        checkpoint_cb = callbacks.ModelCheckpoint(
            f"{model_name}_checkpoint.h5", save_best_only=True
        )
        early_stopping_cb = callbacks.EarlyStopping(
            monitor=self.early_stopping, patience=15
        )
        self.callbacks = [checkpoint_cb, early_stopping_cb]
        return self.callbacks

    def save_model(self, weights=True, output_path="."):
        """The model architecture, and training configuration (including the optimizer, losses, and metrics)
        are stored in saved_model.pb. The weights are saved in the variables/ directory."""
        if self.name is None:
            self.name = str(self.model.name_scope().rstrip("/"))
            datestamp = dt.datetime.now().isoformat().split("T")[0]
            model_name = f"{self.name}_{datestamp}"
        else:
            model_name = self.name

        model_path = os.path.join(output_path, "models", model_name)
        weights_path = f"{model_path}/weights/ckpt"
        self.model.save(model_path)
        if weights is True:
            self.model.save_weights(weights_path)
        for root, _, files in os.walk(model_path):
            indent = "    " * root.count(os.sep)
            print("{}{}/".format(indent, os.path.basename(root)))
            for filename in files:
                print("{}{}".format(indent + "    ", filename))

    def model_diagram(
        self, model=None, fname=None, shapes=True, dtype=False, LR=False, expand=True
    ):
        if model is None:
            model = self.model
        if fname is None:
            fname = model.name
        if LR is True:
            rank = "LR"
        else:
            rank = "TB"
        plot_model(
            model,
            to_file=fname,
            show_shapes=shapes,
            show_dtype=dtype,
            show_layer_names=True,
            rankdir=rank,
            expand_nested=expand,
            dpi=96,
            layer_range=None,
        )

    # def timer(self, func, model_name):
    #     def wrap():
    #         start = time.time()
    #         stopwatch(f"TRAINING ***{model_name}***", t0=start)
    #         func()
    #         end = time.time()
    #         stopwatch(f"TRAINING ***{model_name}***", t0=start, t1=end)
    #         return func
    #     return wrap

    # @timer
    def batch_fit(self):
        """
        Fits cnn and returns keras history
        Gives equal number of positive and negative samples rotating randomly
        """
        model_name = str(self.model.name_scope().rstrip("/").upper())
        print("FITTING MODEL...")
        validation_data = (self.X_test, self.y_test)

        if self.early_stopping is not None:
            self.callbacks = self.set_callbacks()

        start = time.time()
        stopwatch(f"TRAINING ***{model_name}***", t0=start)

        make_batches, steps_per_epoch = self.batch_steps()

        self.history = self.model.fit(
            make_batches,
            validation_data=validation_data,
            verbose=self.verbose,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks,
        )
        end = time.time()
        stopwatch(f"TRAINING ***{model_name}***", t0=start, t1=end)
        self.model.summary()
        return self.history

    def fit(self, params=None):
        """Fit a model to the training data.

        Args:
            params (Dict, optional): [description]. Defaults to None.

        Returns:
            object: keras history (training and validation metrics for each epoch)
        """
        if params is not None:
            self.fit_params(**params)
        model_name = str(self.model.name_scope().rstrip("/").upper())
        print("FITTING MODEL...")
        validation_data = (self.X_test, self.y_test)

        if self.early_stopping is not None:
            self.callbacks = self.set_callbacks()

        start = time.time()
        stopwatch(f"TRAINING ***{model_name}***", t0=start)

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=self.verbose,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )
        end = time.time()
        stopwatch(f"TRAINING ***{model_name}***", t0=start, t1=end)
        self.model.summary()
        return self.history


class MultiLayerPerceptron(Builder):
    def __init__(self, X_train, y_train, X_test, y_test, blueprint="mlp"):
        super().__init__(X_train, y_train, X_test, y_test)
        self.blueprint = blueprint
        self.input_shape = X_train.shape[1]
        self.output_shape = 1
        self.layers = [18, 32, 64, 32, 18]
        self.input_name = "mlp_inputs"
        self.output_name = "mlp_output"
        self.name = "sequential_mlp"
        self.activation = "leaky_relu"
        self.cost_function = "sigmoid"
        self.lr_sched = True
        self.optimizer = Adam
        self.loss = "binary_crossentropy"
        self.metrics = ["accuracy"]

    def build_mlp(self):
        # visible layer
        inputs = Input(shape=(self.input_shape,), name=self.input_name)
        # hidden layers
        x = Dense(
            self.layers[0], activation=self.activation, name=f"1_dense{self.layers[0]}"
        )(inputs)
        for i, layer in enumerate(self.layers[1:]):
            i += 1
            x = Dense(layer, activation=self.activation, name=f"{i+1}_dense{layer}")(x)
        # output layer
        if self.blueprint == "ensemble":
            self.mlp = Model(inputs, x, name="mlp_ensemble")
            return self.mlp
        else:
            self.model = Sequential()
            outputs = Dense(
                self.output_shape, activation=self.cost_function, name=self.output_name
            )(x)
            self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
            if self.lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer(learning_rate=lr_schedule),
                metrics=self.metrics,
            )
            return self.model

    def batch_mlp(self):
        """
        Gives equal number of positive and negative samples rotating randomly
        The output of the generator must be either
        - a tuple `(inputs, targets)`
        - a tuple `(inputs, targets, sample_weights)`.

        This tuple (a single output of the generator) makes a single
        batch. The last batch of the epoch is commonly smaller than the others,
        if the size of the dataset is not divisible by the batch size.
        The generator loops over its data indefinitely.
        An epoch finishes when `steps_per_epoch` batches have been seen by the model.

        """
        # hb: half-batch
        hb = self.batch_size // 2
        # Returns a new array of given shape and type, without initializing.
        xb = np.empty((self.batch_size, self.X_train.shape[1]), dtype="float32")
        # y_train.shape = (2016, 1)
        yb = np.empty((self.batch_size, self.y_train.shape[1]), dtype="float32")

        pos = np.where(self.y_train[:, 0] == 1.0)[0]
        neg = np.where(self.y_train[:, 0] == 0.0)[0]

        # rotating each of the samples randomly
        while True:
            np.random.shuffle(pos)
            np.random.shuffle(neg)

            xb[:hb] = self.X_train[pos[:hb]]
            xb[hb:] = self.X_train[neg[hb : self.batch_size]]
            yb[:hb] = self.y_train[pos[:hb]]
            yb[hb:] = self.y_train[neg[hb : self.batch_size]]

            for i in range(self.batch_size):
                xb[i] = augment_data(xb[i])

            yield xb, yb


class ImageCNN3D(Builder):
    def __init__(self, X_train, y_train, X_test, y_test, blueprint="cnn3d"):
        super().__init__(
            X_train,
            y_train,
            X_test,
            y_test,
        )
        self.blueprint = blueprint
        self.input_shape = self.X_train.shape[1:]
        self.output_shape = 1
        self.data_format = "channels_last"
        self.input_name = "cnn3d_inputs"
        self.output_name = "cnn3d_output"
        self.activation = "leaky_relu"
        self.cost_function = "sigmoid"
        self.loss = "binary_crossentropy"
        self.optimizer = Adam
        self.lr_sched = True
        self.metrics = ["accuracy"]
        self.name = "cnn3d"
        self.kernel = 3
        self.padding = "same"
        self.filters = [32, 64, 128, 256]
        self.pool = 1
        self.dense = 512
        self.dropout = 0.3

    def build_3D(self):
        """Build a 3D convolutional neural network for RGB image triplets"""

        inputs = Input(self.input_shape, name=self.input_name)

        x = Conv3D(
            filters=32,
            kernel_size=self.kernel,
            padding=self.padding,
            data_format=self.data_format,
            activation=self.activation,
        )(inputs)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        for f in self.filters:
            x = Conv3D(
                filters=f,
                kernel_size=self.kernel,
                padding=self.padding,
                activation=self.activation,
            )(x)
            x = MaxPool3D(pool_size=self.pool)(x)
            x = BatchNormalization()(x)

        x = GlobalAveragePooling3D()(x)
        x = Dense(units=self.dense, activation=self.activation)(x)
        x = Dropout(self.dropout)(x)

        if self.blueprint == "ensemble":
            self.cnn = Model(inputs, x, name="cnn_ensemble")
            return self.cnn
        else:
            outputs = Dense(
                units=self.output_shape,
                activation=self.cost_function,
                name=self.output_name,
            )(x)
            if self.lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model = Model(inputs, outputs, name=self.name)
            self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer(learning_rate=lr_schedule),
                metrics=self.metrics,
            )
            return self.model

    def batch_cnn(self):
        """
        Gives equal number of positive and negative samples rotating randomly
        The output of the generator must be either
        - a tuple `(inputs, targets)`
        - a tuple `(inputs, targets, sample_weights)`.

        This tuple (a single output of the generator) makes a single
        batch. The last batch of the epoch is commonly smaller than the others,
        if the size of the dataset is not divisible by the batch size.
        The generator loops over its data indefinitely.
        An epoch finishes when `steps_per_epoch` batches have been seen by the model.

        """
        # hb: half-batch
        hb = self.batch_size // 2
        # Returns a new array of given shape and type, without initializing.

        xb = np.empty(
            (
                self.batch_size,
                self.X_train.shape[1],
                self.X_train.shape[2],
                self.X_train.shape[3],
                self.X_train.shape[4],
            ),
            dtype="float32",
        )

        yb = np.empty((self.batch_size, self.y_train.shape[1]), dtype="float32")

        pos = np.where(self.y_train[:, 0] == 1.0)[0]
        neg = np.where(self.y_train[:, 0] == 0.0)[0]

        # rotating each of the samples randomly
        while True:
            np.random.shuffle(pos)
            np.random.shuffle(neg)

            xb[:hb] = self.X_train[pos[:hb]]
            xb[hb:] = self.X_train[neg[hb : self.batch_size]]
            yb[:hb] = self.y_train[pos[:hb]]
            yb[hb:] = self.y_train[neg[hb : self.batch_size]]

            for i in range(self.batch_size):
                xb[i] = augment_image(xb[i])

            yield xb, yb


class Ensemble(Builder):
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        params=None,
        input_name="svm_mixed_inputs",
        output_name="ensemble_output",
        name="ensembl4D",
    ):
        super().__init__(X_train, y_train, X_test, y_test)
        self.input_name = input_name
        self.output_name = output_name
        self.name = name
        self.blueprint = "ensemble"
        self.ensemble = True
        self.lr_sched = True
        self.params = params
        self.loss = "binary_crossentropy"
        self.optimizer = Adam
        self.metrics = ["accuracy"]
        self.activation = "leaky_relu"
        self.cost_function = "sigmoid"
        # default params:
        self.batch_size = 32
        self.epochs = 1000
        self.lr = 1e-4
        self.decay = [100000, 0.96]
        self.early_stopping = False
        self.verbose = 2
        self.output_shape = 1
        self.layers = [18, 32, 64, 32, 18]
        if params is not None:
            self.fit_params(**params)

    def ensemble_mlp(self):
        self.mlp = MultiLayerPerceptron(
            self.X_train[0],
            self.y_train,
            self.X_test[0],
            self.y_test,
            blueprint="ensemble",
        )
        self.mlp.input_name = "svm_regression_inputs"
        self.mlp.output_name = "svm_regression_output"
        self.mlp.name = "svm_mlp"
        self.mlp.ensemble = True
        self.mlp.input_shape = self.X_train[0].shape[1]
        # default attributes for MLP:
        self.mlp.output_shape = 1
        self.mlp.layers = [18, 32, 64, 32, 18]
        self.mlp.activation = "leaky_relu"
        self.mlp.cost_function = "sigmoid"
        self.mlp.lr_sched = True
        self.mlp.optimizer = Adam
        self.mlp.loss = "binary_crossentropy"
        self.mlp.metrics = ["accuracy"]
        # compile the MLP branch
        self.mlp.model = self.mlp.build_mlp()
        return self.mlp

    def ensemble_cnn(self):
        self.cnn = ImageCNN3D(
            self.X_train[1],
            self.y_train,
            self.X_test[1],
            self.y_test,
            blueprint="ensemble",
        )
        self.cnn.input_name = "svm_image_inputs"
        self.cnn.output_name = "svm_image_output"
        self.cnn.name = "svm_cnn"
        self.cnn.ensemble = True
        self.cnn.input_shape = self.X_train[1].shape[1:]
        # default attributes for CNN:
        self.cnn.output_shape = 1
        self.cnn.layers = [18, 32, 64, 32, 18]
        self.cnn.activation = "leaky_relu"
        self.cnn.cost_function = "sigmoid"
        self.cnn.lr_sched = True
        self.cnn.optimizer = Adam
        self.cnn.loss = "binary_crossentropy"
        self.cnn.metrics = ["accuracy"]
        # compile the CNN branch
        self.cnn.model = self.cnn.build_3D()
        return self.cnn

    def build_ensemble(self):
        self.mlp = self.ensemble_mlp()
        self.cnn = self.ensemble_cnn()
        combinedInput = concatenate([self.mlp.model.output, self.cnn.model.output])
        x = Dense(9, activation="leaky_relu", name=self.input_name)(combinedInput)
        x = Dense(1, activation="sigmoid", name=self.output_name)(x)
        self.model = Model(
            inputs=[self.mlp.model.input, self.cnn.model.input],
            outputs=x,
            name=self.name,
        )
        if self.lr_sched is True:
            lr_schedule = self.decay_learning_rate()
        else:
            lr_schedule = self.lr
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer(learning_rate=lr_schedule),
            metrics=self.metrics,
        )
        return self.model

    def batch_ensemble(self):
        """
        Gives equal number of positive and negative samples rotating randomly
        The output of the generator must be either
        - a tuple `(inputs, targets)`
        - a tuple `(inputs, targets, sample_weights)`.

        This tuple (a single output of the generator) makes a single
        batch. The last batch of the epoch is commonly smaller than the others,
        if the size of the dataset is not divisible by the batch size.
        The generator loops over its data indefinitely.
        An epoch finishes when `steps_per_epoch` batches have been seen by the model.
        """
        # hb: half-batch
        hb = self.batch_size // 2
        # Returns a new array of given shape and type, without initializing.
        xa = np.empty((self.batch_size, self.X_train[0].shape[1]), dtype="float32")
        xb = np.empty(
            (
                self.batch_size,
                self.X_train[1].shape[1],
                self.X_train[1].shape[2],
                self.X_train[1].shape[3],
                self.X_train[1].shape[4],
            ),
            dtype="float32",
        )
        yb = np.empty((self.batch_size, self.y_train.shape[1]), dtype="float32")
        pos = np.where(self.y_train[:, 0] == 1.0)[0]
        neg = np.where(self.y_train[:, 0] == 0.0)[0]
        # rotating each of the samples randomly
        while True:
            np.random.shuffle(pos)
            np.random.shuffle(neg)
            xa[:hb] = self.X_train[0][pos[:hb]]
            xa[hb:] = self.X_train[0][neg[hb : self.batch_size]]
            xb[:hb] = self.X_train[1][pos[:hb]]
            xb[hb:] = self.X_train[1][neg[hb : self.batch_size]]
            yb[:hb] = self.y_train[pos[:hb]]
            yb[hb:] = self.y_train[neg[hb : self.batch_size]]
            for i in range(self.batch_size):
                xa[i] = augment_data(xa[i])
                xb[i] = augment_image(xb[i])
            yield [xa, xb], yb


class Cnn2dBuilder(Builder):
    def __init__(self, X_train, y_train, X_test, y_test, blueprint="cnn2d"):
        super().__init__(self, X_train, y_train, X_test, y_test)
        self.blueprint = blueprint
        self.input_shape = self.X_train.shape[1:]
        self.output_shape = 1
        self.input_name = "cnn2d_inputs"
        self.output_name = "cnn2d_output"
        self.kernel = 11
        self.activation = "relu"
        self.strides = 4
        self.filters = [8, 16, 32, 64]
        self.dense = [64, 64]
        self.dropout = [0.5, 0.25]
        self.optimizer = Adam
        self.lr = 1e-5
        self.lr_sched = False
        self.loss = "binary_crossentropy"
        self.metrics = ["accuracy"]
        self.decay = None
        self.early_stopping = None
        self.batch_size = 32
        self.cost_function = "sigmoid"

    def build_cnn(self):
        """
        Builds and compiles a 2-dimensional Keras Functional CNN
        """
        print("BUILDING MODEL...")
        inputs = Input(self.input_shape, name=self.input_name)
        print("INPUT LAYER")
        x = Conv1D(
            filters=self.filters[0],
            kernel_size=self.kernel,
            activation=self.activation,
            input_shape=self.input_shape,
        )(inputs)
        x = MaxPool1D(strides=self.strides)(x)
        x = BatchNormalization()(x)
        count = 1
        for f in self.filters[1:]:
            x = Conv1D(
                filters=self.filters[f],
                kernel_size=self.kernel,
                activation=self.activation,
            )(x)
            x = MaxPool1D(strides=self.strides)(x)
            if count < len(self.filters):
                x = BatchNormalization()(x)
                count += 1
            else:
                x = Flatten()(x)
        print("DROPOUT")
        for drop, dense in list(zip(self.dropout, self.dense)):
            x = Dropout(drop)(x)
            x = Dense(dense, activation=self.activation)(x)
        if self.blueprint == "ensemble":
            self.cnn = Model(inputs, x, name="cnn2d_ensemble")
            return self.cnn
        else:
            print("FULL CONNECTION")
            outputs = Dense(
                units=self.output_shape,
                activation=self.cost_function,
                name=self.output_name,
            )(x)
            if self.lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model = Model(inputs, outputs, name=self.name)
            self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer(learning_rate=lr_schedule),
                metrics=self.metrics,
            )
            print("COMPILED")
            return self.model

    def batch_maker(self):
        """
        Gives equal number of positive and negative samples rotating randomly
        The output of the generator must be either
        - a tuple `(inputs, targets)`
        - a tuple `(inputs, targets, sample_weights)`.

        This tuple (a single output of the generator) makes a single
        batch. Therefore, all arrays in this tuple must have the same
        length (equal to the size of this batch). Different batches may have
        different sizes.

        For example, the last batch of the epoch is commonly smaller than the others,
        if the size of the dataset is not divisible by the batch size.
        The generator is expected to loop over its data indefinitely.
        An epoch finishes when `steps_per_epoch` batches have been seen by the model.

        """

        # hb: half-batch
        hb = self.batch_size // 2

        # Returns a new array of given shape and type, without initializing.
        # x_train.shape = (5087, 3197, 2)
        xb = np.empty(
            (self.batch_size, self.X_train.shape[1], self.X_train.shape[2]),
            dtype="float32",
        )
        # y_train.shape = (5087, 1)
        yb = np.empty((self.batch_size, self.y_train.shape[1]), dtype="float32")

        pos = np.where(self.y_train[:, 0] == 1.0)[0]
        neg = np.where(self.y_train[:, 0] == 0.0)[0]

        # rotating each of the samples randomly
        while True:
            np.random.shuffle(pos)
            np.random.shuffle(neg)

            xb[:hb] = self.X_train[pos[:hb]]
            xb[hb:] = self.X_train[neg[hb : self.batch_size]]
            yb[:hb] = self.y_train[pos[:hb]]
            yb[hb:] = self.y_train[neg[hb : self.batch_size]]

            for i in range(self.batch_size):
                size = np.random.randint(xb.shape[1])
                xb[i] = np.roll(xb[i], size, axis=0)

            yield xb, yb


""" Pre-fabricated Networks (hyperparameters tuned for specific projects) """


class MemoryClassifier(MultiLayerPerceptron):
    """Builds an MLP neural network classifier object with pre-tuned params for Calcloud's Memory Bin classifier

    Args:
        MultiLayerPerceptron (object): mlp multi-classification builder object
    """

    def __init__(
        self, X_train, y_train, X_test, y_test, blueprint="mlp", test_idx=None
    ):
        super().__init__(X_train, y_train, X_test, y_test, blueprint=blueprint)
        self.input_shape = X_train.shape[1]  # 9
        self.output_shape = 4
        self.layers = [18, 32, 64, 32, 18, 9]
        self.input_name = "hst_jobs"
        self.output_name = "memory_classifier"
        self.name = "mem_clf"
        self.activation = "relu"
        self.cost_function = "softmax"
        self.lr_sched = False
        self.optimizer = Adam
        self.loss = "categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.test_idx = test_idx
        self.epochs = 60
        self.batch_size = 32
        self.algorithm = "multiclass"


class MemoryRegressor(MultiLayerPerceptron):
    """Builds an MLP neural network regressor object with pre-tuned params for estimating memory allocation (GB) in Calcloud

    Args:
        MultiLayerPerceptron (object): mlp linear regression builder object
    """

    def __init__(
        self, X_train, y_train, X_test, y_test, blueprint="mlp", test_idx=None
    ):
        super().__init__(X_train, y_train, X_test, y_test, blueprint=blueprint)
        self.input_shape = X_train.shape[1]  # 9
        self.output_shape = 1
        self.layers = [18, 32, 64, 32, 18, 9]
        self.input_name = "hst_jobs"
        self.output_name = "memory_regressor"
        self.name = "mem_reg"
        self.activation = "relu"
        self.cost_function = "linear"
        self.lr_sched = False
        self.optimizer = Adam
        self.loss = "mse"
        self.metrics = [RMSE(name="rmse")]
        self.test_idx = test_idx
        self.epochs = 60
        self.batch_size = 32
        self.algorithm = "linreg"


class WallclockRegressor(MultiLayerPerceptron):
    """Builds an MLP neural network regressor object with pre-tuned params for estimating wallclock allocation (minimum execution time in seconds) in Calcloud

    Args:
        MultiLayerPerceptron (object): mlp linear regression builder object
    """

    def __init__(
        self, X_train, y_train, X_test, y_test, blueprint="mlp", test_idx=None
    ):
        super().__init__(X_train, y_train, X_test, y_test, blueprint=blueprint)
        self.input_shape = X_train.shape[1]  # 9
        self.output_shape = 1
        self.layers = [18, 32, 64, 128, 256, 128, 64, 32, 18, 9]
        self.input_name = "hst_jobs"
        self.output_name = "wallclock_regressor"
        self.name = "wall_reg"
        self.activation = "relu"
        self.cost_function = "linear"
        self.lr_sched = False
        self.optimizer = Adam
        self.loss = "mse"
        self.metrics = [RMSE(name="rmse")]
        self.test_idx = test_idx
        self.epochs = 300
        self.batch_size = 64
        self.algorithm = "linreg"
