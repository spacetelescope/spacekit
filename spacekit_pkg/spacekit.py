# SPACEKIT
# Machine Learning Utility Functions for Astrophysics
"""
* Radio
    - Radio.mast_aws()
* Analyzer
    - Analyzer.atomic_vector_plotter()
    - Analyzer.make_specgram()
    - Analyzer.planet_hunter()
* Transformer
    - Transfomer.hypersonic_pliers()
    - Transformer.thermo_fusion_chisel()
    - Transformer.babel_fish_dispenser()
    - Transformer.fast_fourier()
* Builder
    - Builder.build_cnn()
    - Builder.fit_cnn()

"""
import pandas as pd
import numpy as np

class Radio:

    # function for downloading data from MAST s3 bucket on AWS
    @staticmethod
    def mast_aws(target_list):
        import boto3
        from astroquery.mast import Observations
        from astroquery.mast import Catalogs
        # configure aws settings
        region = 'us-east-1'
        s3 = boto3.resource('s3', region_name=region)
        bucket = s3.Bucket('stpubdata')
        location = {'LocationConstraint': region}
        Observations.enable_cloud_dataset(provider='AWS', profile='default')
        
        for target in target_list:
        #Do a cone search and find the K2 long cadence data for target
            obs = Observations.query_object(target,radius="0s")
            want = (obs['obs_collection'] == "K2") & (obs['t_exptime'] ==1800.0)
            data_prod = Observations.get_product_list(obs[want])
            filt_prod = Observations.filter_products(data_prod, productSubGroupDescription="LLC")
            s3_uris = Observations.get_cloud_uris(filt_prod)
            for url in s3_uris:
            # Extract the S3 key from the S3 URL
                fits_s3_key = url.replace("s3://stpubdata/", "")
                root = url.split('/')[-1]
                bucket.download_file(fits_s3_key, root, ExtraArgs={"RequestPayer": "requester"})
        Observations.disable_cloud_dataset()
        return print('Download Complete')

class Analyzer:

    def __init__(self, signal):
        self.signal = signal
        self.atomic_vector_plotter = atomic_vector_plotter()
        self.make_specgram = make_specgram()

    @staticmethod
    def atomic_vector_plotter(signal, label_col=None, classes=None, class_names=None, figsize=(15,5), 
    y_units=None, x_units=None):
        """
        Plots scatter and line plots of time series signal values.  
        
        **ARGS
        signal: pandas series or numpy array
        label_col: name of the label column if using labeled pandas series
            -use default None for numpy array or unlabeled series.
            -this is simply for customizing plot Title to include classification    
        classes: (optional- req labeled data) tuple if binary, array if multiclass
        class_names: tuple or array of strings denoting what the classes mean
        figsize: size of the figures (default = (15,5))
        
        ******
        
        Ex1: Labeled timeseries passing 1st row of pandas dataframe
        > first create the signal:
        signal = x_train.iloc[0, :]
        > then plot:
        atomic_vector_plotter(signal, label_col='LABEL',classes=[1,2], 
                    class_names=['No Planet', 'Planet']), figsize=(15,5))
        
        Ex2: numpy array without any labels
        > first create the signal:
        signal = x_train.iloc[0, :]

        >then plot:
        atomic_vector_plotter(signal, figsize=(15,5))
        
        ######
        TODO: 
        -`signal` should take an array rather than pdseries
        -could allow either array or series to be passed, conv to array if series 
        ######
        """
        import pandas as pd
        import numpy as np

        ### LABELS
        # pass None to label_col if unlabeled data, creates generic title
        if label_col is None:
            label = None
            title_scatter = "Scatterplot of Star Flux Signals"
            title_line = "Line Plot of Star Flux Signals"
            color='black'
            
        # store target column as variable 
        elif label_col is not None:
            label = signal[label_col]
            # for labeled timeseries
            if label == 1:
                cn = class_names[0]
                color='red'

            elif label == 2:
                cn = class_names[1] 
                color='blue'
            ## TITLES
        #create appropriate title acc to class_names    
            title_scatter = f"Scatterplot for Star Flux Signal: {cn}"
            title_line = f"Line Plot for Star Flux Signal: {cn}"
        
        # Set x and y axis labels according to units
        # if the units are unknown, we will default to "Flux"
        if y_units == None:
            y_units = 'Flux'
        else:
            y_units = y_units
        # it is assumed this is a timeseries, default to "time"   
        if x_units == None:
            x_units = 'Time'
        else:
            x_units = x_units
        
        # Scatter Plot
        
        if type(signal) == array:
            series_index=list(range(len(signal)))

            converted_array = pd.Series(signal.ravel(), index=series_index)
            signal = converted_array 
        
        plt.figure(figsize=figsize)
        plt.scatter(pd.Series([i for i in range(1, len(signal))]), 
                    signal[1:], marker=4, color=color)
        plt.ylabel(y_units)
        plt.xlabel(x_units)
        plt.title(title_scatter)
        plt.show()

        # Line Plot
        plt.figure(figsize=figsize)
        plt.plot(pd.Series([i for i in range(1, len(signal))]), 
                signal[1:], color=color)
        plt.ylabel(y_units)
        plt.xlabel(x_units)
        plt.title(title_line)
        plt.show()

    ### MAKE_SPECGRAM
    # create function for generating and saving spectograph figs
    @staticmethod
    def make_specgram(signal, Fs=None, NFFT=None, noverlap=None, mode=None,
                    cmap=None, units=None, colorbar=False, 
                    save_for_ML=False, fname=None,num=None,**kwargs):
        import matplotlib
        import matplotlib.pyplot as plt
        if mode:
            mode=mode
        if cmap is None:
            cmap='binary'

        #PIX: plots only the pixelgrids -ideal for image classification
        if save_for_ML == True:
            # turn off everything except pixel grid
            fig, ax = plt.subplots(figsize=(10,10),frameon=False)
            fig, freqs, t, m = plt.specgram(signal, Fs=Fs, NFFT=NFFT, mode=mode,cmap=cmap)
            ax.axis(False)
            ax.show()

            if fname is not None:
                try:
                    if num:
                        path=fname+num
                    else:
                        path=fname
                    plt.savefig(path,**pil_kwargs)
                except:
                    print('Something went wrong while saving the img file')

        else:
            fig, ax = plt.subplots(figsize=(13,11))
            fig, freqs, t, m = plt.specgram(signal, Fs=Fs, NFFT=NFFT, mode=mode,cmap=cmap)
            plt.colorbar()
            if units is None:
                units=['Wavelength (λ)','Frequency (ν)']
            plt.xlabel(units[0])
            plt.ylabel(units[1])
            if num:
                title=f'Spectogram_{num}'
            else:
                title='Spectogram'
            plt.title(title)
            plt.show()

        return fig, freqs, t, m

    def planet_hunter(f, fmt='kepler.fits'):
        """
        args:
        - fits_files = takes array or single .fits file
        
        kwargs:
        - format : 'kepler.fits' or  'tess.fits'
        Ex:
        planet_hunter(f=files[9], fmt='kepler.fits')
        """
        from astropy.timeseries import TimeSeries
        import numpy as np
        from astropy import units as u
        from astropy.timeseries import BoxLeastSquares
        from astropy.stats import sigma_clipped_stats
        from astropy.timeseries import aggregate_downsample

        # read in file
        ts = TimeSeries.read(f, format=fmt)
        
    #     print(f'mjd: {ts.time.mjd[0]}, {ts.time.mjd [-1]}')
    #     # tdb = Barycentric Dynamical Time scale
    #     print(f'time_scale: {ts.time.scale}')
    #     print(ts['time', 'sap_flux'])
        
        
        # original ts plot
        #fig, ax = plt.figure()
    #    plt.plot(ts.time.jd, ts['sap_flux'], 'k.', markersize=1)
        
    #     plt.xlabel('Julian Date')
    #     plt.ylabel('SAP Flux (e-/s)')
        
        # use box least squares to estimate period
        periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')
        results = periodogram.autopower(0.2 * u.day)  
        best = np.argmax(results.power)  
        period = results.period[best]  
        transit_time = results.transit_time[best]  
        print(f'transit_time: {transit_time}')
        
        # fold the time series using the period 
        ts_folded = ts.fold(period=period, epoch_time=transit_time) 
        
    #     # folded time series plot
    #     plt.plot(ts_folded.time.jd, ts_folded['sap_flux'], 'k.', markersize=1)
    #     plt.xlabel('Time (days)')
    #     plt.ylabel('SAP Flux (e-/s)')
        
        #normalize the flux by sigma-clipping the data to determine the baseline flux:
        
        mean, median, stddev = sigma_clipped_stats(ts_folded['sap_flux'])  
        ts_folded['sap_flux_norm'] = ts_folded['sap_flux'] / median  

        # downsample the time series by binning the points into bins of equal time 
        # - this returns a BinnedTimeSeries:
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.03 * u.day)  

    #     # final result
        fig = plt.figure(figsize=(11,5))
        plt.plot(ts_folded.time.jd, ts_folded['sap_flux_norm'], 'k.', markersize=1)
        plt.plot(ts_binned.time_bin_start.jd, ts_binned['sap_flux_norm'], 'r-', drawstyle='steps-post')
        plt.xlabel('Time (days)')
        plt.ylabel('Normalized flux')



class Transformer:

    
    @staticmethod
    def hypersonic_pliers(path_to_train, path_to_test):
        
        """
        Using Numpy to extract data into 1-dimensional arrays
        separate target classes (y) for training and test data
        assumes y (target) is first column in dataframe
        
        #TODO: option to pass in column index loc for `y` if not default (0) 
        #TODO: option for `y` to already be 0 or 1 (don't subtract 1)
        #TODO: allow option for `y` to be categorical (convert via binarizer)
        """
        import numpy as np
        
        Train = np.loadtxt(path_to_train, skiprows=1, delimiter=',')
        X_train = Train[:, 1:]
        y_train = Train[:, 0, np.newaxis] - 1.
        
        Test = np.loadtxt(path_to_test, skiprows=1, delimiter=',')
        X_test = Test[:, 1:]
        y_test = Test[:, 0, np.newaxis] - 1.
        
        del Train,Test
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        
        return X_train, X_test, y_train, y_test


    @staticmethod
    def thermo_fusion_chisel(matrix1, matrix2=None):
            """
            Scales each array of a matrix to zero mean and unit variance.
            returns matrix/matrices of same shape as input but scaled
            matrix2 is optional - useful if data was already train-test split
            example: matrix1=X_train, matrix2=X_test
            
            """
            import numpy as np
            
                
            matrix1 = ((matrix1 - np.mean(matrix1, axis=1).reshape(-1,1)) / 
                np.std(matrix1, axis=1).reshape(-1,1))
            
            print("Mean: ",matrix1[0].mean())
            print("Variance: ",matrix1[0].std())
            
            if matrix2 is not None:
                matrix2 = ((matrix2 - np.mean(matrix2, axis=1).reshape(-1,1)) / 
                    np.std(matrix2, axis=1).reshape(-1,1))
            
                print("Mean: ",matrix2[0].mean())
                print("Variance: ",matrix2[0].std())
                return matrix1,matrix2
            else:
                return matrix1


    @staticmethod
    def babel_fish_dispenser(matrix1, matrix2=None, step_size=None, axis=2):
            """        
            Adds an input corresponding to the running average over a set number
            of time steps. This helps the neural network to ignore high frequency 
            noise by passing in a uniform 1-D filter and stacking the arrays. 
            
            **ARGS
            step_size: integer, # timesteps for 1D filter. defaults to 200
            axis: which axis to stack the arrays
            
            ex:
            noise_filter(matrix1=X_train, matrix2=X_test, step_size=200)
            """
            import numpy as np
            from scipy.ndimage.filters import uniform_filter1d
            
            if step_size is None:
                step_size=200
                
            # calc input for flux signal rolling avgs 
            filter1 = uniform_filter1d(matrix1, axis=1, size=step_size)
            # store in array and stack on 2nd axis for each obs of X data
            matrix1 = np.stack([matrix1, filter1], axis=2)
            
            if matrix2 is not None:
                filter2 = uniform_filter1d(matrix2, axis=1, size=step_size)
                matrix2 = np.stack([matrix2, filter2], axis=2)
                print(matrix1.shape,matrix2.shape)
                return matrix1,matrix2
            else:
                print(matrix1.shape)
                return matrix1


    @staticmethod
    def fast_fourier(matrix, bins):
        """
        takes in array and rotates #bins to the left as a fourier transform
        returns vector of length equal to input array
        """
        import numpy as np

        shape = matrix.shape
        fourier_matrix = np.zeros(shape, dtype=float)
        
        for row in matrix:
            signal = np.asarray(row)
            frequency = np.arange(signal.size/2+1, dtype=np.float)
            phase = np.exp(complex(0.0, (2.0*np.pi)) * frequency * bins / float(signal.size))
            ft = np.fft.irfft(phase * np.fft.rfft(signal))
            fourier_matrix += ft
        
        return fourier_matrix
        
        
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
        import keras
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



import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, jaccard_score, fowlkes_mallows_score
from sklearn.metrics import confusion_matrix, roc_curve

class Computer:

    @staticmethod
    def get_preds(X,y,model=None,verbose=False):
        if model is None:
            model=model
        # class predictions 
        y_true = y.flatten()
        
        y_pred = model.predict_classes(X).flatten() 
        preds = pd.Series(y_pred).value_counts(normalize=False)
        
        if verbose:
            print(f"y_pred:\n {preds}")
            print("\n")

        return y_true, y_pred

    @staticmethod
    def fnfp(X,y,model, training=False):

        import numpy as np

        y_pred = np.round( model.predict(X) )

        pos_idx = y==1
        neg_idx = y==0

        #tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
        fn = np.sum(y_pred[pos_idx]==0)/y_pred.shape[0]

        #tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
        fp = np.sum(y_pred[neg_idx]==1)/y_pred.shape[0]

        if training:
            print(f"FN Rate (Training): {round(fn*100,4)}%")
            print(f"FP Rate (Training): {round(fp*100,4)}%")
        else:
            print(f"FN Rate (Test): {round(fn*100,4)}%")
            print(f"FP Rate (Test): {round(fp*100,4)}%")

    @staticmethod
    def keras_history(history, figsize=(10,4)):
        """
        side by side sublots of training val accuracy and loss (left and right respectively)
        """
        
        import matplotlib.pyplot as plt
        
        fig,axes=plt.subplots(ncols=2,figsize=(15,6))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')

        ax = axes[1]
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    # FUSION_MATRIX()
    @staticmethod 
    def fusion_matrix(matrix, classes=None, normalize=True, title='Fusion Matrix', cmap='Blues',
        print_raw=False): 
        """
        FUSION MATRIX!
        -------------
        It's like a confusion matrix...without the confusion.
        
        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
        classes: class names for target variables
        """
        from sklearn import metrics                       
        from sklearn.metrics import confusion_matrix #ugh
        import itertools
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        # make matrix if tuple passed to matrix:
        if isinstance(matrix, tuple):
            y_true = matrix[0].copy()
            y_pred = matrix[1].copy()
            
            if y_true.ndim>1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim>1:
                y_pred = y_pred.argmax(axis=1)
            fusion = metrics.confusion_matrix(y_true, y_pred)
        else:
            fusion = matrix
        
        # INTEGER LABELS
        if classes is None:
            classes=list(range(len(matrix)))

        #NORMALIZING
        # Check if normalize is set to True
        # If so, normalize the raw fusion matrix before visualizing
        if normalize:
            fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
            thresh = 0.5
            fmt='.2f'
        else:
            fmt='d'
            thresh = fusion.max() / 2.
        
        # PLOT
        fig, ax = plt.subplots(figsize=(10,10))
        plt.imshow(fusion, cmap=cmap, aspect='equal')
        
        # Add title and axis labels 
        plt.title(title) 
        plt.ylabel('TRUE') 
        plt.xlabel('PRED')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #ax.set_ylim(len(fusion), -.5,.5) ## <-- This was messing up the plots!
        
        # Text formatting
        fmt = '.2f' if normalize else 'd'
        # Add labels to each cell
        #thresh = fusion.max() / 2.
        # iterate thru matrix and append labels  
        for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
            plt.text(j, i, format(fusion[i, j], fmt),
                    horizontalalignment='center',
                    color='white' if fusion[i, j] > thresh else 'black',
                    size=14, weight='bold')
        
        # Add a legend
        plt.colorbar()
        plt.show() 
        return fusion, fig

    @staticmethod
    def roc_plots(X,y,model):
        """Calculates ROC_AUC score and plots Receiver Operator Characteristics (ROC)

        Arguments:
            X {feature set} -- typically X_test
            y {labels} -- typically y_test
            model {classifier} -- the model name for which you are calculting roc score

        Returns:
            roc -- roc_auc_score (via sklearn)
        """
        from sklearn import metrics
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

        y_true = y.flatten()
        y_hat = model.predict(X)

        fpr, tpr, thresholds = roc_curve(y_true, y_hat) 

        # Threshold Cutoff for predictions
        crossover_index = np.min(np.where(1.-fpr <= tpr))
        crossover_cutoff = thresholds[crossover_index]
        crossover_specificity = 1.-fpr[crossover_index]

        fig,axes=plt.subplots(ncols=2, figsize=(15,6))
        axes = axes.flatten()

        ax=axes[0]
        ax.plot(thresholds, 1.-fpr)
        ax.plot(thresholds, tpr)
        ax.set_title("Crossover at {0:.2f}, Specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

        ax=axes[1]
        ax.plot(fpr, tpr)
        ax.set_title("ROC area under curve: {0:.2f}".format(roc_auc_score(y_true, y_hat)))
        plt.show()
        
        roc = roc_auc_score(y_true,y_hat)

        return roc


        ### COMPUTE SCORES ###
    @staticmethod
    def compute(X, y, model, hist=None, preds=True, summary=True, fusion=True, 
                classes=None, report=True, roc=True):
        """
        evaluates model predictions and stores the output in a dict
        returns `results`
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn import metrics
        from sklearn.metrics import jaccard_score,accuracy_score, recall_score, fowlkes_mallows_score

        # initialize a spare improbability drive
        res = {}
        res['model'] = model.name
        
        # class predictions 
        if preds:
            y_true = y.flatten()
            y_pred = model.predict_classes(X).flatten()
            res['preds'] = [y_pred]

        if summary:
            summary = model.summary()
            res['summary'] = model.summary


        # FUSION MATRIX
        if fusion:
            if classes is None:
                classes=set(y)
                #classes=['0','1']
            else:
                classes=classes
            # Plot fusion matrix
            FM = fusion_matrix(matrix=(y_true,y_pred), 
                                        classes=classes)
            res['FM'] = FM

        # ROC Area Under Curve
        if roc:
            ROC = roc_plots(X, y, model)
            res['ROC'] = ROC

        # CLASSIFICATION REPORT
        if report:
            num_dashes=20
            print('\n')
            print('---'*num_dashes)
            print('\tCLASSIFICATION REPORT:')
            print('---'*num_dashes)
            # generate report
            report = metrics.classification_report(y_true,y_pred)
            res['report'] = report
            print(report)


        # save to dict:
        res['jaccard'] = jaccard_score(y_true, y_pred)
        res['fowlkes'] = fowlkes_mallows_score(y_true,y_pred)
        res['accuracy'] = accuracy_score(y_true, y_pred)
        res['recall'] = recall_score(y_true, y_pred)
        
        #Plot Model Training Results (PLOT KERAS HISTORY)
        if hist is not None:
            HIST = keras_history(hist)
            res['HIST'] = HIST

        return res



if __name__ == '__main__':
    radio = Radio()
    radio.mast_aws()
    analyzer = Analyzer()
    analyzer.atomic_vector_plotter()
    analyzer.make_specgram()
    analyzer.planet_hunter()
    transformer = Transformer()
    transformer.hypersonic_pliers()
    transformer.thermo_fusion_chisel()
    transformer.babel_fish_dispenser()
    transformer.fast_fourier()
    builder = Builder()
    builder.build_cnn()
    builder.fit_cnn()
    computer = Computer()
    computer.get_preds()
    computer.fnfp()
    computer.keras_history()
    computer.fusion_matrix()
    computer.roc_plots()
    computer.compute()