
# ********* starskøpe.Spacekit.signal ********* #
"""
analyzer
helper functions for signal and wave analysis  

"""
# -----------------
# STATIC CLASS METHODS 
# -----------------
# * atomic_vector_plotter() 
# * make_specgram()
# * planet_hunter()
# 
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #

import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

### Flux Signals

@staticmethod
def atomic_vector_plotter(signal, label_col=None, classes=None, class_names=None, figsize=(15,5), y_units=None, x_units=None):
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
    
    if type(signal) == np.array:
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


@staticmethod
def make_specgram(signal, Fs=2, NFFT=256, noverlap=128, mode='psd',
                cmap=None, units=None, colorbar=False, 
                save_for_ML=False, fname=None,num=None,**kwargs):
    """generate and save spectographs of flux signal frequencies
    """
    import matplotlib
    import matplotlib.pyplot as plt

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
                plt.savefig(path,**kwargs)
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
            title=f'Spectrogram_{num}'
        else:
            title='Spectrogram'
        plt.title(title)
        plt.show()

    return fig, freqs, t, m

@staticmethod
def planet_hunter(file_list, fmt='kepler.fits', error=False, snr=False):
    """plots phase-folded light curve of a signal
    returns dataframe of transit timestamps for each light curve 
    planet_hunter(f=files[9], fmt='kepler.fits')
    
    args:
    - fits_files = takes array of files or single .fits file
    
    kwargs:
    - format : 'kepler.fits' or  'tess.fits'
    - error: include SAP flux error (residuals) if available
    - snr: apply signal-to-noise-ratio to periodogram autopower calculation
    """
    from astropy.timeseries import TimeSeries
    import numpy as np
    from astropy import units as u
    from astropy.timeseries import BoxLeastSquares
    from astropy.stats import sigma_clipped_stats
    from astropy.timeseries import aggregate_downsample
    
    # read in file
    transits = {}
    for index, file in enumerate(file_list):
        res = {}
        if fmt == 'kepler.fits':
            prefix = file.replace('ktwo','') 
            suffix = prefix.replace('_llc.fits','')
            pair = suffix.split('-')
            obs_id = pair[0]
            campaign = pair[1]
        
        ts = TimeSeries.read(file, format=fmt) # read in timeseries
        
        # add to meta dict
        res['obs_id'] = obs_id
        res['campaign'] = campaign
        res['lc_start'] = ts.time.jd[0]
        res['lc_end'] = ts.time.jd[-1]

        # use box least squares to estimate period
        if error is True: # if error col data available
            periodogram = BoxLeastSquares.from_timeseries(ts,'sap_flux','sap_flux_err')
        else:
            periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')
        if snr is True:
            results = periodogram.autopower(0.2 * u.day, objective='snr')
        else:
            results = periodogram.autopower(0.2 * u.day)
        
        maxpower = np.argmax(results.power)  
        period = results.period[maxpower] 
        transit_time = results.transit_time[maxpower]
        
        res['maxpower'] = maxpower
        res['period'] = period
        res['transit'] = transit_time
        
        #res['ts'] = ts
        
        # fold the time series using the period 
        ts_folded = ts.fold(period=period, epoch_time=transit_time)
        
        # folded time series plot
        # plt.plot(ts_folded.time.jd, ts_folded['sap_flux'], 'k.', markersize=1)
        # plt.xlabel('Time (days)')
        # plt.ylabel('SAP Flux (e-/s)')

        #normalize the flux by sigma-clipping the data to determine the baseline flux:
        mean, median, stddev = sigma_clipped_stats(ts_folded['sap_flux'])
        ts_folded['sap_flux_norm'] = ts_folded['sap_flux'] / median 
        res['mean'] = mean
        res['median'] = median
        res['stddev'] = stddev
        res['sap_flux_norm'] = ts_folded['sap_flux_norm']
        
        # downsample the time series by binning the points into bins of equal time 
        ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.03 * u.day)  

        # final result
        fig = plt.figure(figsize=(11,5))
        ax = fig.gca()
        ax.plot(ts_folded.time.jd, ts_folded['sap_flux_norm'], 'k.', markersize=1)
        ax.plot(ts_binned.time_bin_start.jd, ts_binned['sap_flux_norm'], 'r-', drawstyle='steps-post')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized flux')
        ax.set_title(obs_id)
        ax.legend([np.round(period, 3)])
        plt.close()
        
        res['fig'] = fig
        
        transits[index] = res
        
    df = pd.DataFrame.from_dict(transits, orient='index')
        
    return df
