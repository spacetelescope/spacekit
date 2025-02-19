import os
import sys
import importlib.resources
from zipfile import ZipFile
import numpy as np
import time
import datetime as dt
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
from spacekit.analyzer.track import stopwatch, xtimer
from spacekit.builder.blueprints import Blueprint
from spacekit.logger.log import Logger

TF_CPP_MIN_LOG_LEVEL = 2


class Builder:
    """Class for building and training a neural network."""

    def __init__(
        self,
        train_data=None,
        test_data=None,
        blueprint=None,
        algorithm=None,
        model_path=None,
        name=None,
        logname="Builder",
        **log_kws,
    ):
        if train_data is not None:
            self.X_train = train_data[0]
            self.y_train = train_data[1]
        if test_data is not None:
            self.X_test = test_data[0]
            self.y_test = test_data[1]
        self.blueprint = blueprint
        self.algorithm = algorithm
        self.model_path = model_path
        self.name = name
        self.batch_size = 32
        self.epochs = 60
        self.lr = 1e-4
        self.decay = [100000, 0.96]
        self.early_stopping = None
        self.patience = 15
        self.callbacks = None
        self.verbose = 2
        self.ensemble = False
        self.model = None
        self.mlp = None
        self.cnn = None
        self.ann = None
        self.strides = None
        self.kernel_size = None
        self.step_size = None
        self.steps_per_epoch = None
        self.batch_maker = None
        self.history = None
        self.tx_file = None
        self.__name__ = logname
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()

    def load_saved_model(
        self,
        arch=None,
        compile_params=None,
        custom_obj={},
        extract_to="models",
        keras_archive=True,
    ):
        """Load saved keras model from local disk (located at the ``model_path`` attribute) or a pre-trained model from 
        spacekit.skopes.trained_networks (if ``model_path`` attribute is None). Example for ``compile_params``:
        ``dict(loss="binary_crossentropy",metrics=["accuracy"],\
        optimizer=Adam(learning_rate=optimizers.schedules.ExponentialDecay(1e-4, \
        decay_steps=100000, decay_rate=0.96, staircase=True)))``

        Parameters
        ----------
        arch : str, optional
            select a pre-trained model from the spacekit library ("svm_align", "jwst_cal", or "hst_cal"), by default None
        compile_params : dict, optional
            Compile the model using kwarg parameters, by default None
        custom_obj : dict, optional
            custom objects keyword arguments to be passed into Keras `load_model`, by default {}
        extract_to : str or path, optional
            directory location to extract into, by default "models"
        keras_archive : bool, optional
            for models saved in the newer high-level .keras compressed archive format (recommended), by default True

        Returns
        -------
        Keras functional Model object
            pre-trained (and/or compiled) functional Keras model.
        """
        if self.model_path is None:
            self.load_pretrained_network(arch=arch)
        if str(self.model_path).split(".")[-1] == "zip":
            self.unzip_model_files(extract_to=extract_to)
        model_basename = os.path.basename(self.model_path.rstrip("/"))
        if model_basename != self.name:  # for spacekit archives, this is always True
            for root, dirs, files in os.walk(self.model_path):
                if keras_archive is True:
                    for f in files:
                        if f == f"{self.name}.keras":
                            self.model_path = os.path.join(root, f)
                            break
                else:  # for legacy SavedModel folder containing assets, variables and saved_model.pb
                    for d in dirs:
                        if d == self.name:
                            self.model_path = os.path.join(root, d)

        self.log.debug(f"Loading saved model from {str(self.model_path)}")
        try:
            self.model = load_model(self.model_path, custom_objects=custom_obj)
            if compile_params:
                self.model = Model(inputs=self.model.inputs, outputs=self.model.outputs)
                self.model.compile(**compile_params)
        except Exception as e:
            self.log.error(e)
        return self.model

    def load_pretrained_network(self, arch=None):
        mission_blueprints = ["hst", "jwst"]
        err = None
        if arch is None:
            err = "Must specify spacekit pretrained NN using `arch` when model_path attribute is not set."
        elif arch not in ["hst_cal", "jwst_cal", "svm_align"]:
            err = f"The spacekit pretrained NN archive named {arch} does not exist."
        if err:
            self.log.error(err)
            sys.exit(1)
        model_src = "spacekit.builder.trained_networks"
        archive_file = f"{arch}.zip"  # hst_cal.zip | jwst_cal.zip | svm_align.zip
        with importlib.resources.path(model_src, archive_file) as mod:
            self.model_path = mod
        if self.blueprint is None:
            mission_arch = arch.split("_")[0]
            if mission_arch in mission_blueprints:
                self.blueprint = f"{mission_arch}_{self.name}"
            elif mission_arch == "svm":
                self.blueprint = "ensemble"

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

    def find_tx_file(self, name="tx_data.json"):
        if not self.model_path:
            return
        for root, _, files in os.walk(os.path.dirname(self.model_path)):
            for f in files:
                if f == name:
                    self.tx_file = os.path.join(root, f)
                    break
        if not os.path.exists(self.tx_file):
            self.tx_file = os.path.join(self.model_path, name)

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
        algorithm=None,
    ):
        """Set custom build parameters for a Builder object.

        Parameters
        ----------
        input_shape : tuple, optional
            shape of the inputs, by default None
        output_shape : tuple, optional
            shape of the output, by default None
        layers: list, optional
            sizes of hidden (dense) layers, by default None
        kernel_size : int, optional
            size of the kernel, by default None
        activation : string, optional
            dense layer activation, by default None
        cost_function: str, optional
            function to update weights (calculate cost), by default None
        strides : int, optional
            number of strides, by default None
        optimizer : object, optional
            type of optimizer to use, by default None
        lr_sched: bool, optional
            use a learning_rate schedule such as ExponentialDecay, by default None
        loss : string, optional
            loss metric to monitor, by default None
        metrics : list, optional
            metrics for model to train on, by default None
        algorithm : str, optional
            analysis type, used for determining spacekit.analyzer.Compute class e.g. "linreg"
            for linear regression or "multiclass" for multi-label classification, by default None

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
        self.algorithm = algorithm
        self.input_name = input_name
        self.output_name = output_name
        self.name = name
        return self

    def get_build_params(self):
        """Build params for initialized model

        Returns
        -------
        dict
            dictionary of key-val pairs for initialized model's build parameters
        """
        return dict(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            layers=self.layers,
            kernel_size=self.kernel_size,
            activation=self.activation,
            cost_function=self.cost_function,
            strides=self.strides,
            optimizer=self.optimizer,
            lr_sched=self.lr_sched,
            loss=self.loss,
            metrics=self.metrics,
            input_name=self.input_name,
            output_name=self.output_name,
            name=self.name,
            algorithm=self.algorithm
        )

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

    def get_fit_params(self):
        """Fit parameters of built model

        Returns
        -------
        dict
            key-val pairs of model fitting hyperparameters
        """
        return dict(
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            decay=self.decay,
            early_stopping=self.early_stopping,
            verbose=self.verbose,
            ensemble=self.ensemble,
        )

    def get_blueprint(self, architecture, fitting=True):
        draft = Blueprint(architecture=architecture)
        self.set_build_params(**draft.building())
        if fitting is True:
            self.fit_params(**draft.fitting())
        return draft

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

    def set_callbacks(self):
        """Set an early stopping callback by monitoring the model training for either
        accuracy or loss.  For classifiers, use 'val_accuracy' or 'val_loss'.
        For regression use 'val_loss' or 'val_rmse'. Default patience is 15
        and can be modified by setting the `self.patience` attribute to a custom value (int).

        Returns
        -------
        list
            [callbacks.ModelCheckpoint, callbacks.EarlyStopping]
        """
        if self.early_stopping not in ['val_accuracy', 'val_loss', 'val_rmse']:
            warn_msg = f"Invalid `early_stopping` value: {self.early_stopping}."
            if self.algorithm == "linreg" or self.loss == "mse":
                self.early_stopping = 'val_rmse'
            elif "accuracy" in self.metrics:
                self.early_stopping = 'val_accuracy'
            else:
                self.early_stopping = 'val_loss'
            self.log.warning(f"{warn_msg} Using default : {self.early_stopping}")
        model_name = str(self.model.name)
        checkpoint_cb = callbacks.ModelCheckpoint(
            f"{model_name}_checkpoint.keras", save_best_only=True
        )
        early_stopping_cb = callbacks.EarlyStopping(
            monitor=self.early_stopping, patience=self.patience
        )
        self.callbacks = [checkpoint_cb, early_stopping_cb]
        return self.callbacks

    def save_keras_model(self, model_path):
        dpath = os.path.dirname(model_path)
        os.makedirs(dpath, exist_ok=True)
        name = os.path.basename(model_path)
        if not name.endswith("keras"):
            name += ".keras"
        keras_model_path = os.path.join(dpath, name)
        self.model.save(keras_model_path)
        self.model_path = keras_model_path

    def save_model(self, output_path=".", keras_archive=True, parent_dir="", weights=True):
        """The model architecture, and training configuration (including the optimizer, losses, and metrics)
        are stored in saved_model.pb. The weights are saved in the variables/ directory.

        Parameters
        ----------
        output_path : str, optional
            where to save the model files, by default "."
        keras_archive : bool, optional
            save model using new (preferred) keras archive format, by default True
        parent_dir : str, optional
            store model in a subdirectory of output_path using this name, by default ""
        weights : bool, optional
            save weights learned by the model separately also, by default True
        """
        if self.name is None:
            self.name = str(self.model.name)
            datestamp = dt.datetime.now().isoformat().split("T")[0]
            model_name = f"{self.name}_{datestamp}"
        else:
            model_name = self.name

        if str(os.path.basename(output_path)).lower() == "models":
            model_path = os.path.join(output_path, parent_dir, model_name)
        else:
            model_path = os.path.join(output_path, "models", parent_dir, model_name)

        if keras_archive is True:
            self.save_keras_model(model_path)
        else:
            self.model.save(model_path)
            if weights is True:
                weights_path = f"{model_path}/weights/ckpt.keras"
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
            from tensorflow.keras.utils import plot_model  # req: pydot, graphviz

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
        except ImportError:
            self.log.error(
                "pydot and graphviz not installed: `pip install spacekit[viz]`"
            )

    @xtimer
    def batch_fit(self):
        """
        Fits cnn using a batch generator of equal positive and negative number of samples, rotating randomly.

        Returns
        -------
        tf.keras.model.history
            Keras training history
        """
        model_name = str(self.model.name).upper()
        self.log.info("FITTING MODEL...")
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

    @xtimer
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
        model_name = str(self.model.name).upper()
        self.log.info("FITTING MODEL...")
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

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        blueprint="mlp",
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            logname="BuilderMLP",
            **builder_kwargs,
        )
        self.blueprint = blueprint
        self.name = "sequential_mlp"
        self.input_shape = X_train.shape[1] if X_train is not None else None
        self.output_shape = 1
        self.layers = [18, 32, 64, 32, 18]
        self.input_name = "mlp_inputs"
        self.output_name = "mlp_output"
        self.activation = "leaky_relu"
        self.cost_function = "sigmoid"
        self.lr_sched = True
        self.optimizer = Adam
        self.loss = "binary_crossentropy"
        self.metrics = ["accuracy"]
        self.step_size = X_train.shape[0] if X_train is not None else self.batch_size
        self.steps_per_epoch = self.step_size // self.batch_size
        self.batch_maker = self.batch

    def build(self, layer_kwargs={}):
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
            if layer in layer_kwargs:
                x = Dense(layer, activation=self.activation, name=f"{i+1}_dense{layer}", **layer_kwargs[layer])(x)
            else:
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

    def batch(self, augment=True):
        """Randomly rotates through positive and negative samples indefinitely,
        generating a single batch tuple of (inputs, targets) or (inputs, targets, sample_weights).
        If the size of the dataset is not divisible by the batch size, the last batch will be
        smaller than the others. An epoch finishes once `steps_per_epoch` have been seen by the model.

        Parameters
        ----------
        augment : bool, optional
            whether or not to apply stochastic data augmentation on input features, by default True

        Yields
        ------
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

            if augment is True:
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

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        blueprint="cnn3d",
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            logname="BuilderCNN3D",
            **builder_kwargs,
        )
        self.blueprint = blueprint
        self.input_shape = X_train.shape[1:] if X_train is not None else None
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
        self.step_size = (
            self.X_train.shape[0] if self.X_train is not None else self.batch_size
        )
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
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        params=None,
        input_name="svm_mixed_inputs",
        output_name="ensemble_output",
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            logname="BuilderEnsemble",
            **builder_kwargs,
        )
        self.name = "ensembleSVM"
        self.input_name = input_name
        self.output_name = output_name
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
        self.step_size = X_train[0].shape[0] if X_train is not None else self.batch_size
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
            X_train=self.X_train[0],
            y_train=self.y_train,
            X_test=self.X_test[0],
            y_test=self.y_test,
            blueprint="ensemble",
        )
        # experimental (sets all attrs below)
        # self.mlp.get_blueprint("svm_mlp")
        self.mlp.input_name = "svm_regression_inputs"
        self.mlp.output_name = "svm_regression_output"
        self.mlp.name = "svm_mlp"
        self.mlp.ensemble = True
        self.mlp.input_shape = self.X_train[0].shape[1] if self.X_train else None
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
            X_train=self.X_train[1],
            y_train=self.y_train,
            X_test=self.X_test[1],
            y_test=self.y_test,
            blueprint="ensemble",
        )
        # self.cnn.get_blueprint("svm_cnn")
        self.cnn.input_name = "svm_image_inputs"
        self.cnn.output_name = "svm_image_output"
        self.cnn.name = "svm_cnn"
        self.cnn.ensemble = True
        self.cnn.input_shape = self.X_train[1].shape[1:] if self.X_train is not None else None
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
        x = Dense(9, activation=self.activation, name=self.input_name)(combinedInput)
        x = Dense(1, activation=self.cost_function, name=self.output_name)(x)
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
            yield (xa, xb), yb


class BuilderCNN2D(Builder):
    """Subclass Builder object for 2D Convolutional Neural Networks

    Parameters
    ----------
    Builder : class
        spacekit.builder.architect.Builder.Builder object
    """

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        blueprint="cnn2d",
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            logname="BuilderCNN2D",
            **builder_kwargs,
        )
        self.blueprint = blueprint
        self.input_shape = self.X_train.shape[1:] if self.X_train is not None else None
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
        self.step_size = X_train.shape[1] if X_train is not None else None
        self.steps_per_epoch = self.step_size // self.batch_size
        self.batch_maker = self.batch

    def build(self):
        """
        Builds and compiles a 2-dimensional Keras Functional CNN
        """
        self.log.info("BUILDING MODEL...")
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
        for f in list(range(len(self.filters))):
            if f == 0:
                continue
            x = Conv1D(
                filters=self.filters[f],
                kernel_size=self.kernel,
                activation=self.activation,
            )(x)
            x = MaxPool1D(strides=self.strides)(x)
            if f < len(self.filters) - 1:
                x = BatchNormalization()(x)
            else:
                x = Flatten()(x)
        self.log.info("DROPOUT")
        for drop, dense in list(zip(self.dropout, self.dense)):
            x = Dropout(drop)(x)
            x = Dense(dense, activation=self.activation)(x)
        if self.blueprint == "ensemble":
            self.cnn = Model(inputs, x, name="cnn2d_ensemble")
            return self.cnn
        else:
            self.log.info("FULL CONNECTION")
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
            self.log.info("COMPILED")
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
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        blueprint="hst_mem_clf",
        test_idx=None,
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            blueprint=blueprint,
            algorithm="multiclass",
            logname="MemoryClassifier",
            **builder_kwargs,
        )
        self.input_shape = self.X_train.shape[1] if self.X_train else 9
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


class MemoryRegressor(BuilderMLP):
    """Builds an MLP neural network regressor object with pre-tuned params for estimating memory allocation (GB) in Calcloud

    Args:
        MultiLayerPerceptron (object): mlp linear regression builder object
    """

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        blueprint="hst_mem_reg",
        test_idx=None,
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            blueprint=blueprint,
            algorithm="linreg",
            logname="MemoryRegressor",
            **builder_kwargs,
        )
        self.input_shape = self.X_train.shape[1] if self.X_train else 9
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


class WallclockRegressor(BuilderMLP):
    """Builds an MLP neural network regressor object with pre-tuned params for estimating
    wallclock allocation (minimum execution time in seconds) in Calcloud.

    Args:
        MultiLayerPerceptron (object): mlp linear regression builder object
    """

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        blueprint="hst_wall_reg",
        test_idx=None,
        **builder_kwargs,
    ):
        train_data = (
            (X_train, y_train) if X_train is not None and y_train is not None else None
        )
        test_data = (
            (X_test, y_test) if X_test is not None and y_test is not None else None
        )
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            blueprint=blueprint,
            algorithm="linreg",
            logname="WallclockRegressor",
            **builder_kwargs,
        )
        self.input_shape = self.X_train.shape[1] if self.X_train else 9
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
