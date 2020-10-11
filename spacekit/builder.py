# ********* starskøpe.spacekit.builder ********* #
"""
* Builder
    - Builder.build_cnn()
    - Builder.fit_cnn()

#TODO:
    * build_lstm()
    * fit_lstm
    * class RBM
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models, layers, optimizers
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #

class Builder:

    @staticmethod
    def build_cnn(X_train, X_test, y_train, y_test, kernel_size=None, activation=None, input_shape=None, strides=None, optimizer=Adam, 
                    learning_rate=None, loss=None, 
                    metrics=None):
        """
        Builds and compiles linear CNN using Keras

        """
        from tensorflow import keras
        from keras.utils.np_utils import to_categorical
        # from keras.preprocessing.text import Tokenizer
        from keras import models, layers, optimizers
        from keras.models import Sequential, Model
        from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
        BatchNormalization, Input, concatenate, Activation
        from keras.optimizers import Adam
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import cross_val_score

        if input_shape is None:
            input_shape = X_train.shape[1:]
        if kernel_size is None:
            kernel_size=11
        if activation is None:
            activation='relu'
        if strides is None:
            strides = 4
        if learning_rate is None:
            learning_rate = 1e-5
        if loss is None:
            loss='binary_crossentropy'
        if metrics is None:
            metrics=['accuracy']
        
        print("BUILDING MODEL...")
        model=Sequential()
        
        print("LAYER 1")
        model.add(Conv1D(filters=8, kernel_size=kernel_size, 
                        activation=activation, input_shape=input_shape))
        model.add(MaxPool1D(strides=strides))
        model.add(BatchNormalization())
        
        print("LAYER 2")
        model.add(Conv1D(filters=16, kernel_size=kernel_size, 
                        activation=activation))
        model.add(MaxPool1D(strides=strides))
        model.add(BatchNormalization())
        
        print("LAYER 3")
        model.add(Conv1D(filters=32, kernel_size=kernel_size, 
                        activation=activation))
        model.add(MaxPool1D(strides=strides))
        model.add(BatchNormalization())
        
        print("LAYER 4")
        model.add(Conv1D(filters=64, kernel_size=kernel_size, 
                        activation=activation))
        model.add(MaxPool1D(strides=strides))
        model.add(Flatten())
        
        print("FULL CONNECTION")
        model.add(Dropout(0.5))
        model.add(Dense(64, activation=activation))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation=activation))

        print("ADDING COST FUNCTION")
        model.add(Dense(1, activation='sigmoid'))
 
        ##### COMPILE #####
        model.compile(optimizer=optimizer(learning_rate), loss=loss, 
                    metrics=metrics)
        print("COMPILED")  
        
        return model 


    @staticmethod
    def fit_cnn(X_train,y_train, X_test, y_test, model, validation_data=None, 
                verbose=None, epochs=None, steps_per_epoch=None, batch_size=None):
        """
        Fits cnn and returns keras history
        Gives equal number of positive and negative samples rotating randomly  
        """
        if verbose is None:
            verbose=2
        if epochs is None:
            epochs = 5
        if validation_data is None:
            validation_data=(X_test, y_test)
        if steps_per_epoch is None:
            steps_per_epoch = (X_train.shape[1]//batch_size)
        if batch_size is None:
            batch_size = 32

        print("FITTING MODEL...")
        
        def batch_maker(X_train, y_train, batch_size=batch_size):
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
                import numpy
                import random
                # hb: half-batch
                hb = batch_size // 2
                
                # Returns a new array of given shape and type, without initializing.
                # x_train.shape = (5087, 3197, 2)
                xb = np.empty((batch_size, X_train.shape[1], X_train.shape[2]), dtype='float32')
                
                #y_train.shape = (5087, 1)
                yb = np.empty((batch_size, y_train.shape[1]), dtype='float32')
                
                pos = np.where(y_train[:,0] == 1.)[0]
                neg = np.where(y_train[:,0] == 0.)[0]

                # rotating each of the samples randomly
                while True:
                    np.random.shuffle(pos)
                    np.random.shuffle(neg)
                
                    xb[:hb] = X_train[pos[:hb]]
                    xb[hb:] = X_train[neg[hb:batch_size]]
                    yb[:hb] = y_train[pos[:hb]]
                    yb[hb:] = y_train[neg[hb:batch_size]]
                
                    for i in range(batch_size):
                        size = np.random.randint(xb.shape[1])
                        xb[i] = np.roll(xb[i], size, axis=0)
                
                    yield xb, yb
        
        history = model.fit_generator(batch_maker(X_train, y_train, batch_size),
                                        validation_data=validation_data, 
                                        verbose=verbose, epochs=epochs, 
                                        steps_per_epoch=steps_per_epoch)
        print("TRAINING COMPLETE")
        model.summary()

        return history


######## Bidirectional CNN #####
# predicts a probability of each type of class

    # @staticmethod
    # def build_lstm(X_train, X_test, y_train, y_test, kernel_size=None, activation=None, input_shape=None, strides=None, optimizer=Adam, 
    #                 learning_rate=None, loss=None, 
    #                 metrics=None):
    #     """
    #     Builds and compiles a bidirectional RNN using Keras

    #     """
    #     import pandas as pd
    #     import numpy as np
    #     np.random.seed(0)
    #     import matplotlib.pyplot as plt
    #     %matplotlib inline
    #     import keras
    #     from keras.preprocessing.sequence import pad_sequences
    #     from keras.layers import Input, Dense, LSTM, Embedding
    #     from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
    #     from keras.models import Model
    #     from keras import initializers, regularizers, constraints, optimizers, layers
    #     from keras.preprocessing import text, sequence
    #     from keras.callbacks import ModelCheckpoint
    #     from keras.callbacks import EarlyStopping
        
    #     embedding_size = 128

    #     model = Sequential()
    #     model.add(Bidirectional(LSTM(10, return_sequences=True), 
    #                             input_shape=(100,)))
        # model.add(Bidirectional(LSTM(10, return_sequences=True),
        #                         input_shape=(5, 10)))
        # model.add(Bidirectional(LSTM(10)))
        # model.add(Dense(5))
        # model.add(Activation('softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  

        # return model

    # @staticmethod
    # def fit_lstm(X_train,y_train, X_test, y_test, model, validation_data=None, 
    #             verbose=None, epochs=None, steps_per_epoch=None, batch_size=None):
    #     #     * `X_t`
    #     #     * `y`
    #     #     * `batch_size=32`
    #     #     * `epochs=1`
    #     #     * `validation_split=0.1`
    #     #     * `callbacks=callbacks`

    #     # checkpoints_path = 'weights_base.best.hdf5'
    #     # checkpoint = ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    #     # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=25)
    #     # callbacks = [checkpoint, early_stopping]

    #     history = model.fit_generator(batch_maker(X_train, y_train, batch_size),
    #                                     validation_data=validation_data, 
    #                                     verbose=verbose, epochs=epochs, 
    #                                     steps_per_epoch=steps_per_epoch)
    #     print("TRAINING COMPLETE")
    #     model.summary()
        
    #     return history
        
        
        # stellar_properties = np.array('solar_mass','solar_radius','solar_metallicity','temp_k')
        # stellar_classes = np.array('red_giant', 'hot_jupiter', 'mini_neptune','super_earth','earth_habitable'
        # tce_classes = np.array('possible_planet', 'eclipsing_binary','artifact')

        # y = 
        # # slice `list_classes` from train, and then chain it with `.values`). Store this in `y`.
        # stellar_properties = np.array('solar_mass','solar_radius','solar_metallicity','temp_k')
        # Store the `.values` of `train['comment_text]` in `list_sentences_train`

        #tokenizer = None
        #list_tokenized_train = None
        #X_t = None

        # inputs = known_planets
        # validation = candidates

# class RBM():
#     def __init__(self, nv, nh):
#         self.W = torch.randn(nh, nv)
#         self.a = torch.randn(1, nh)
#         self.b = torch.randn(1, nv)
#     def sample_h(self, x):
#         wx =torch.mm(x, self.W.t())
#         activation = wx + self.a.expand_as(wx)
#         p_h_given_v = torch.sigmoid(activation)
#         return p_h_given_v, torch.bernouilli(p_h_given_v)
#     def sample_v(self, y):
#         wy = torch.mm(y, self.W)
#         activation = wy + self.b.expand_as(wy)
#         p_v_given_h = torch.sigmoid(activation)
#         return p_v_given_h, torch.bernouilli(p_v_given_h)
#     def train(self, v0, vk, ph0, phk):
#         self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
#         self.b += torch.sum((v0 - vk), 0)
#         self.a += torch.sum((ph0 - phk), 0)

    # nv = len(training_set[0])
    # nh = 100
    # batch_size = 100
    # rbm = RBM(nv, nh)

## Training the RBM
# nb_epoch = 10
# for epoch in range(1, nb_epoch +1):
#     train_loss = 0
#     s = 0.
#     for user_id in range(0, nb_users - batch_size, batch_size):
#         vk = training_set[user_id:user_id+batch_size]
#         v0 = training_set[user_id:user_id+batch_size]