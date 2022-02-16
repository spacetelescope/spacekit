import os
import importlib.resources
from zipfile import ZipFile
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
from spacekit.builder.blueprints import Blueprint

TF_CPP_MIN_LOG_LEVEL = 2

# TODO: K-fold cross-val class


class Builder:
    """Class for building and training a neural network."""

    def __init__(
        self, train_data=None, test_data=None, blueprint=None, model_path=None
    ):
        if train_data is not None:
            self.X_train = train_data[0]
            self.y_train = train_data[1]
        if test_data is not None:
            self.X_test = test_data[0]
            self.y_test = test_data[1]
        self.blueprint = blueprint
        self.model_path = model_path
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
        self.step_size = None
        self.steps_per_epoch = None
        self.batch_maker = None
        self.history = None
        self.name = None
        self.tx_file = None

    def load_saved_model(self, arch="ensembleSVM", compile_params=None):
        """Load saved keras model from local disk (located at the ``model_path`` attribute) or a pre-trained model from spacekit.skopes.trained_networks (if ``model_path`` attribute is None). Example for ``compile_params``: ``dict(loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=optimizers.schedules.ExponentialDecay(lr=1e-4, decay_steps=100000, decay_rate=0.96, staircase=True)))``

        Parameters
        ----------
        arch : str, optional
            select a pre-trained model included from the spacekit library of trained networks ("ensembleSVM" or "calmodels"), by default "ensembleSVM"
        compile_params : dict, optional
            Compile the model using kwarg parameters, by default None

        Returns
        -------
        Keras functional Model object
            pre-trained (and/or compiled) functional Keras model.
        """
        if self.model_path is None:
            model_src = "spacekit.builder.trained_networks"
            archive_file = f"{arch}.zip"
            with importlib.resources.path(model_src, archive_file) as mod:
                self.model_path = mod
        if str(self.model_path).split(".")[-1] == "zip":
            self.model_path = self.unzip_model_files()
        self.model = load_model(self.model_path)
        if compile_params:
            self.model = Model(inputs=self.model.inputs, outputs=self.model.outputs)
            self.model.compile(**compile_params)
        return self.model

    def unzip_model_files(self, extract_to="models"):
        """Extracts a keras model object from a zip archive

        Parameters
        ----------
        extract_to : str, optional
            directory location to extract into, by default "models"

        Returns
        -------
        string
            path to where the model archive has been extracted
        """
        os.makedirs(extract_to, exist_ok=True)
        model_base = os.path.basename(self.model_path).split(".")[0]
        with ZipFile(self.model_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        self.model_path = os.path.join(extract_to, model_base)
        return self.model_path

    def find_tx_file(self, name="tx_data.json"):
        if self.model_path:
            tx_file = os.path.join(self.model_path, name)
        if os.path.exists(tx_file):
            self.tx_file = tx_file

    def set_build_params(
        self,
        input_shape=None,
        output_shape=None,
        layers=None,
        kernel_size=None,
        activation=None,
        cost_function=None,
        strides=None,
        optimizer=None,
        lr_sched=None,
        loss=None,
        metrics=None,
        input_name=None,
        output_name=None,
        name=None,
    ):
        """Set custom build parameters for a Builder object.

        Parameters
        ----------
        input_shape : tuple
            shape of the inputs
        output_shape : tuple
            shape of the output
        layers: list
            sizes of hidden (dense) layers
        kernel_size : int
            size of the kernel
        activation : string
            dense layer activation
        cost_function: str
            function to update weights (calculate cost)
        strides : int
            number of strides
        optimizer : object
            type of optimizer to use
        lr_sched: bool
            use a learning_rate schedule such as ExponentialDecay
        loss : string
            loss metric to monitor
        metrics : list
            metrics for model to train on

        Returns
        -------
        self
            spacekit.builder.architect.Builder class object with updated attributes
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.kernel_size = kernel_size
        self.activation = activation
        self.cost_function = cost_function
        self.strides = strides
        self.optimizer = optimizer
        self.lr_sched = lr_sched
        self.loss = loss
        self.metrics = metrics
        self.input_name = input_name
        self.output_name = output_name
        self.name = name
        return self

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
        """Set custom model fitting parameters as Builder object attributes.

        Parameters
        ----------
        batch_size : int, optional
            size of each training batch, by default 32
        epochs : int, optional
            number of epochs, by default 60
        lr : float, optional
            initial learning rate, by default 1e-4
        decay : list, optional
            decay_steps, decay_rate, by default [100000, 0.96]
        early_stopping : str, optional
            use an early stopping callback, by default None
        verbose : int, optional
            set the verbosity level, by default 2
        ensemble : bool, optional
            ensemble type network, by default False

        Returns
        -------
        self
            spacekit.builder.architect.Builder class object with updated fitting parameters.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.decay = decay
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.ensemble = ensemble
        return self

    def get_blueprint(self, architecture, fitting=True):
        draft = Blueprint(architecture=architecture)
        self.set_build_params(**draft.building())
        if fitting is True:
            self.fit_params(**draft.fitting())
        return self

    def decay_learning_rate(self):
        """Set learning schedule with exponential decay

        Returns
        -------
        keras.optimizers.schedules.ExponentialDecay
            exponential decay learning rate schedule
        """
        lr_schedule = optimizers.schedules.ExponentialDecay(
            self.lr, decay_steps=self.decay[0], decay_rate=self.decay[1], staircase=True
        )
        return lr_schedule

    def set_callbacks(self, patience=15):
        """Set an early stopping callback by monitoring the model training for either accuracy or loss.  For classifiers, use 'val_accuracy' or 'val_loss'. For regression use 'val_loss' or 'val_rmse'.

        Returns
        -------
        list
            [callbacks.ModelCheckpoint, callbacks.EarlyStopping]
        """
        model_name = str(self.model.name_scope().rstrip("/"))
        checkpoint_cb = callbacks.ModelCheckpoint(
            f"{model_name}_checkpoint.h5", save_best_only=True
        )
        early_stopping_cb = callbacks.EarlyStopping(
            monitor=self.early_stopping, patience=patience
        )
        self.callbacks = [checkpoint_cb, early_stopping_cb]
        return self.callbacks

    def save_model(self, weights=True, output_path="."):
        """The model architecture, and training configuration (including the optimizer, losses, and metrics)
        are stored in saved_model.pb. The weights are saved in the variables/ directory.

        Parameters
        ----------
        weights : bool, optional
            save weights learned by the model separately also, by default True
        output_path : str, optional
            where to save the model files, by default "."
        """
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
        self.model_path = model_path

    def model_diagram(
        self,
        model=None,
        output_path=None,
        show_shapes=True,
        show_dtype=False,
        LR=False,
        expand_nested=True,
        show_layer_names=False,
    ):
        rank = "LR" if LR is True else "TB"
        if model is None:
            model = self.model
        if output_path is None:
            output_path = os.getcwd()
        try:
            plot_model(
                model,
                to_file=f"{output_path}/{model.name}.png",
                show_shapes=show_shapes,
                show_dtype=show_dtype,
                show_layer_names=show_layer_names,
                rankdir=rank,
                expand_nested=expand_nested,
                dpi=96,
                layer_range=None,
            )
        # TODO error handling
        except Exception as e:
            print(e)

    # TODO
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
        Fits cnn using a batch generator of equal positive and negative number of samples, rotating randomly.

        Returns
        -------
        tf.keras.model.history
            Keras training history
        """
        model_name = str(self.model.name_scope().rstrip("/").upper())
        print("FITTING MODEL...")
        validation_data = (
            (self.X_test, self.y_test) if self.X_test is not None else None
        )

        if self.early_stopping is not None:
            self.callbacks = self.set_callbacks()

        start = time.time()
        stopwatch(f"TRAINING ***{model_name}***", t0=start)

        if self.steps_per_epoch is None or 0:
            self.steps_per_epoch = 1

        self.history = self.model.fit(
            self.batch_maker(),
            validation_data=validation_data,
            verbose=self.verbose,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=self.callbacks,
        )
        end = time.time()
        stopwatch(f"TRAINING ***{model_name}***", t0=start, t1=end)
        self.model.summary()
        return self.history

    def fit(self, params=None):
        """Fit a model to the training data.

        Parameters
        ----------
        params : dictionary, optional
            set custom fit params, by default None

        Returns
        -------
        tf.keras.model.history
            Keras training history object
        """
        if params is not None:
            self.fit_params(**params)
        model_name = str(self.model.name_scope().rstrip("/").upper())
        print("FITTING MODEL...")
        validation_data = (
            (self.X_test, self.y_test) if self.X_test is not None else None
        )

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


class BuilderMLP(Builder):
    """Subclass for building and training MLP neural networks

    Parameters
    ----------
    Builder : class
        spacekit.builder.architect.Builder class object
    """

    def __init__(self, X_train, y_train, X_test, y_test, blueprint="mlp"):
        super().__init__(train_data=(X_train, y_train), test_data=(X_test, y_test))
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
        self.step_size = X_train.shape[0]
        self.steps_per_epoch = self.step_size // self.batch_size
        self.batch_maker = self.batch

    def build(self):
        """Build and compile an MLP network

        Returns
        -------
        tf.keras.model
            compiled model object
        """
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

    def batch(self):
        """Randomly rotates through positive and negative samples indefinitely, generating a single batch tuple of (inputs, targets) or (inputs, targets, sample_weights). If the size of the dataset is not divisible by the batch size, the last batch will be smaller than the others. An epoch finishes once `steps_per_epoch` have been seen by the model.

        Yields
        -------
        tuple
            a single batch tuple of (inputs, targets) or (inputs, targets, sample_weights).
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


class BuilderCNN3D(Builder):
    """Subclass for building and training 3D convolutional neural networks

    Parameters
    ----------
    Builder : class
        spacekit.builder.architect.Builder class object
    """

    def __init__(self, X_train, y_train, X_test, y_test, blueprint="cnn3d"):
        super().__init__(train_data=(X_train, y_train), test_data=(X_test, y_test))
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
        self.step_size = X_train.shape[0]
        self.steps_per_epoch = self.step_size // self.batch_size
        self.batch_maker = self.batch

    def build(self):
        """Builds a 3D convolutional neural network for RGB image triplets"""

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

    def batch(self):
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


class BuilderEnsemble(Builder):
    """Subclass for building and training an ensemble model (stacked MLP and 3D CNN)

    Parameters
    ----------
    Builder : class
        spacekit.builder.architect.Builder class object
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        params=None,
        input_name="svm_mixed_inputs",
        output_name="ensemble_output",
        name="ensembleSVM",
    ):
        super().__init__(train_data=(X_train, y_train), test_data=(X_test, y_test))
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
        self.layers = [18, 32, 64, 32, 18]
        self.output_shape = 1
        # fit params:
        self.batch_size = 32
        self.epochs = 1000
        self.lr = 1e-4
        self.decay = [100000, 0.96]
        self.early_stopping = False
        self.verbose = 2
        if params is not None:
            self.fit_params(**params)
        self.step_size = X_train[0].shape[0]
        self.steps_per_epoch = self.step_size // self.batch_size
        self.batch_maker = self.batch

    def ensemble_mlp(self):
        """Compiles the MLP branch of the ensemble model

        Returns
        -------
        self
            spacekit.builder.architect.Builder.BuilderEnsemble object with ``mlp`` attribute initialized
        """
        self.mlp = BuilderMLP(
            self.X_train[0],
            self.y_train,
            self.X_test[0],
            self.y_test,
            blueprint="ensemble",
        )
        # experimental (sets all attrs below)
        # self.mlp.get_blueprint("svm_mlp")
        self.mlp.input_name = "svm_regression_inputs"
        self.mlp.output_name = "svm_regression_output"
        self.mlp.name = "svm_mlp"
        self.mlp.ensemble = True
        self.mlp.input_shape = self.X_train[0].shape[1]
        self.mlp.output_shape = 1
        self.mlp.layers = [18, 32, 64, 32, 18]
        self.mlp.activation = "leaky_relu"
        self.mlp.cost_function = "sigmoid"
        self.mlp.lr_sched = True
        self.mlp.optimizer = Adam
        self.mlp.loss = "binary_crossentropy"
        self.mlp.metrics = ["accuracy"]
        self.mlp.model = self.mlp.build()
        return self.mlp

    def ensemble_cnn(self):
        """Compiles the CNN branch of the ensemble model

        Returns
        -------
        self
            spacekit.builder.architect.Builder.BuilderEnsemble object with ``cnn`` attribute initialized
        """
        self.cnn = BuilderCNN3D(
            self.X_train[1],
            self.y_train,
            self.X_test[1],
            self.y_test,
            blueprint="ensemble",
        )
        # self.cnn.get_blueprint("svm_cnn")
        self.cnn.input_name = "svm_image_inputs"
        self.cnn.output_name = "svm_image_output"
        self.cnn.name = "svm_cnn"
        self.cnn.ensemble = True
        self.cnn.input_shape = self.X_train[1].shape[1:]
        self.cnn.output_shape = 1
        self.cnn.layers = [18, 32, 64, 32, 18]
        self.cnn.activation = "leaky_relu"
        self.cnn.cost_function = "sigmoid"
        self.cnn.lr_sched = True
        self.cnn.optimizer = Adam
        self.cnn.loss = "binary_crossentropy"
        self.cnn.metrics = ["accuracy"]
        self.cnn.model = self.cnn.build()
        return self.cnn

    def build(self):
        """Builds and compiles the ensemble model

        Returns
        -------
        self
            spacekit.builder.architect.Builder.BuilderEnsemble object with ``model`` attribute initialized
        """
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

    def batch(self):
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


class BuilderCNN2D(Builder):
    """Subclass Builder object for 2D Convolutional Neural Networks

    Parameters
    ----------
    Builder : class
        spacekit.builder.architect.Builder.Builder object
    """

    def __init__(self, X_train, y_train, X_test, y_test, blueprint="cnn2d"):
        super().__init__(train_data=(X_train, y_train), test_data=(X_test, y_test))
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
        self.step_size = X_train.shape[1]
        self.steps_per_epoch = self.step_size // self.batch_size
        self.batch_maker = self.batch

    def build(self):
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

    def batch(self):
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


class MemoryClassifier(BuilderMLP):
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


class MemoryRegressor(BuilderMLP):
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


class WallclockRegressor(BuilderMLP):
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
