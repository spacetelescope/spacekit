# STANDARD libraries
import numpy as np
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.layers import Dense, Input, concatenate
from keras.layers import Dense
# from preprocessor.transform import apply_power_transform, power_transform_matrix
from preprocessor.augment import augment_data, augment_image
from tracker.stopwatch import clocklog

DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2


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
        self.parameters = None
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
        self.history = None
        self.name = None
        self.model_path = None

    def load_saved_model(self):
        model = load_model(self.model_path)
        self.model = Model(inputs=model.inputs, outputs=model.outputs)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.decay_learning_rate()),
            metrics=["accuracy"],
        )
        return self
    
    # def set_parameters(self, params=None):
    #     if params is None:
    #         params = dict(
    #             batch_size=32,
    #             epochs=60,
    #             lr=1e-4,
    #             decay=[100000, 0.96],
    #             early_stopping=None,
    #             verbose=2,
    #             ensemble=True)
    #     return params

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
        early_stopping: 'val_accuracy' or 'val_loss'
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

    def fit_generator(self):
        """
        Fits cnn and returns keras history
        Gives equal number of positive and negative samples rotating randomly
        """
        model_name = str(self.model.name_scope().rstrip("/").upper())
        print(f"FITTING MODEL...")
        validation_data = (self.X_test, self.y_test)

        if self.early_stopping is not None:
            self.callbacks = self.set_callbacks()

        start = time.time()
        clocklog(f"TRAINING ***{model_name}***", t0=start)
        if self.ensemble is True:
            make_batches = self.batch_ensemble()
            steps_per_epoch = self.X_train[0].shape[0] // self.batch_size
        else:
            make_batches = self.batch_maker()
            steps_per_epoch = self.X_train.shape[0] // self.batch_size

        self.history = self.model.fit(
            make_batches,
            validation_data=validation_data,
            verbose=self.verbose,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.callbacks,
        )
        end = time.time()
        clocklog(f"TRAINING ***{model_name}***", t0=start, t1=end)
        self.model.summary()
        return self.history


class MultiLayerPerceptron(Builder):

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(self, X_train, X_test, y_train, y_test)


    def build_mlp(self, input_shape=None, lr_sched=True, layers=[18, 32, 64, 32, 18]):
        if input_shape is None:
            input_shape = self.X_train.shape[1]
        self.model = Sequential()
        # visible layer
        inputs = Input(shape=(input_shape,), name="svm_inputs")
        # hidden layers
        x = Dense(layers[0], activation="leaky_relu", name=f"1_dense{layers[0]}")(
            inputs
        )
        for i, layer in enumerate(layers[1:]):
            i += 1
            x = Dense(layer, activation="leaky_relu", name=f"{i+1}_dense{layer}")(x)
        # output layer
        if self.ensemble is True:
            self.mlp = Model(inputs, x, name="mlp_ensemble")
            return self.mlp
        else:
            outputs = Dense(1, activation="sigmoid", name="svm_output")(x)
            self.model = Model(inputs=inputs, outputs=outputs, name="sequential_mlp")
            if lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=lr_schedule),
                metrics=["accuracy"],
            )
            return self.model


class ImageCNN3D(Builder):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(self, X_train, X_test, y_train, y_test)

    def build_cnn(self, input_shape=None, lr_sched=True):
        """Build a 3D convolutional neural network for RGB image triplets"""
        if input_shape is None:
            input_shape = self.X_train.shape[1:]

        inputs = Input(input_shape, name="img3d_inputs")

        x = layers.Conv3D(
            filters=32,
            kernel_size=3,
            padding="same",
            data_format="channels_last",
            activation="leaky_relu",
        )(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(
            filters=32, kernel_size=3, padding="same", activation="leaky_relu"
        )(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(
            filters=64, kernel_size=3, padding="same", activation="leaky_relu"
        )(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(
            filters=128, kernel_size=3, padding="same", activation="leaky_relu"
        )(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(
            filters=256, kernel_size=3, padding="same", activation="leaky_relu"
        )(x)
        x = layers.MaxPool3D(pool_size=1)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="leaky_relu")(x)
        x = layers.Dropout(0.3)(x)

        if self.ensemble is True:
            self.cnn = Model(inputs, x, name="cnn_ensemble")
            return self.cnn
        else:
            outputs = layers.Dense(units=1, activation="sigmoid", name="img3d_output")(
                x
            )
            # Define the model.
            if lr_sched is True:
                lr_schedule = self.decay_learning_rate()
            else:
                lr_schedule = self.lr
            self.model = Model(inputs, outputs, name="cnn3d")
            self.model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=lr_schedule),
                metrics=["accuracy"],
            )
            return self.model

    def batch_maker(self):
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
        # x_train.shape = (2016, 3, 128, 128, 3)
        if len(self.X_train.shape) == 2:
            xb = np.empty((self.batch_size, self.X_train.shape[1]), dtype="float32")
            augmenter = augment_data
        else:
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
            augmenter = augment_image

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
                xb[i] = augmenter(xb[i])

            yield xb, yb


class Ensemble(Builder):

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(self, X_train, X_test, y_train, y_test)

    def build_ensemble(self, lr_sched=True):
        if self.name is None:
            self.name = "ensemble4d"
        self.mlp = self.build_mlp(input_shape=self.X_train[0].shape[1])
        self.cnn = self.build_cnn(input_shape=self.X_train[1].shape[1:])
        combinedInput = concatenate([self.mlp.output, self.cnn.output])
        x = Dense(9, activation="leaky_relu", name="combined_input")(combinedInput)
        x = Dense(1, activation="sigmoid", name="ensemble_output")(x)
        self.model = Model(
            inputs=[self.mlp.input, self.cnn.input], outputs=x, name=self.name
        )
        if lr_sched is True:
            lr_schedule = self.decay_learning_rate()
        else:
            lr_schedule = self.lr
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr_schedule),
            metrics=["accuracy"],
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
        # x_train.shape = (2016, 3, 128, 128, 3)

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

        # y_train.shape = (2016, 1)
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