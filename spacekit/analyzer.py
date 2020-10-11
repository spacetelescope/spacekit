
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
class Analyzer:

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
    # generate and save spectographs of flux signal frequencies
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

    def planet_hunter(f, fmt='kepler.fits'):
        """
        args:
        - fits_files = takes array or single .fits file
        
        kwargs:
        - format : 'kepler.fits' or  'tess.fits'

        ex:
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
