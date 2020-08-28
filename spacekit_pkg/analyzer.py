
# ********* starskøpe.Spacekit.signal ********* #
"""
analyzer
helper functions for signal and wave analysis  

"""
# -----------------
# STATIC CLASS METHODS 
# -----------------
# * Flux 
#       signal_plots() 
#
# 
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #

import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

### Flux Signals
### 
class Flux:

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
        star_signal_alpha = x_train.iloc[0, :]
        > then plot:
        star_signals(star_signal_alpha, label_col='LABEL',classes=[1,2], 
                    class_names=['No Planet', 'Planet']), figsize=(15,5))
        
        Ex2: numpy array without any labels
        > first create the signal:
        
        >then plot:
        star_signals(signal, figsize=(15,5))
        
        ######
        TODO: 
        -`signal` should take an array rather than pdseries
        -could allow either array or series to be passed, conv to array if series 
        ######
        """
        
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


# todo
# estimate_period
# --> BoxCox
# --> Lomb-Scargle

    @staticmethod
    def phase_bin(time,flux,per,tmid=0,cadence=16,offset=0.25):
        '''
            Phase fold data and bin according to time cadence
            time - [days]
            flux - arbitrary unit
            per - period in [days]
            tmid - value in days
            cadence - spacing to bin data to [minutes] 
        '''
        phase = ((time-tmid)/per + offset)%1

        sortidx = np.argsort(phase)
        sortflux = flux[sortidx]
        sortphase = phase[sortidx]

        cad = cadence/60./24/per # phase cadence according to kepler cadence
        pbins = np.arange(0,1+cad,cad) # phase bins
        bindata = np.zeros(pbins.shape[0]-1)
        for i in range(pbins.shape[0]-1):
            pidx = (sortphase > pbins[i]) & (sortphase < pbins[i+1])

            if pidx.sum() == 0 or np.isnan(sortflux[pidx]).all():
                bindata[i] = np.nan
                continue

            bindata[i] = np.nanmean(sortflux[pidx])

        phases = pbins[:-1]+np.diff(pbins)*0.5

        # remove nans
        #nonans = ~np.isnan(bindata)
        #return phases[nonans],bindata[nonans]
        return phases, bindata




# check for change in relative flux vs position 
    # also subtract off bad pixels 
    #dflux = pdcflux[~bad_data]/np.nanmedian(pdcflux[~bad_data])


### IN PROG 
# Python3 code to demonstrate 
# Every Kth index Maximum in List 
# using list comprehension + enumerate() + max() 

# initializing list 
#test_list = [1, 4, 5, 6, 7, 8, 9, 12] 

# printing the original list 
# print ("The original list is : " + str(test_list)) 

# initializing K 
# K = 3

# using list comprehension + enumerate() + max() 
# Every Kth index Maximum in List 
# max of every 3rd element 
# res = max([i for j, i in enumerate(test_list) if j % K == 0 ]) 

# printing result 
# print ("The max of every kth element : " + str(res)) 



    # # SIGNAL_SPLITS #### WORK IN PROGRESS
    # @staticmethod
    # def signal_splits(signal, figsize=(21,11), **subplot_kws):

    #     fig, axes = plt.subplots(ncols=3, figsize=figsize, **subplot_kws)
    #     axes = axes.flatten()
