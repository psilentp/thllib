import numpy as np 

def idx_by_thresh(signal,thresh = 0.1):
    import numpy as np
    idxs = np.squeeze(np.argwhere(signal > thresh))
    try:
        split_idxs = np.squeeze(np.argwhere(np.diff(idxs) > 1))
    except IndexError:
        #print 'IndexError'
        return None
    #split_idxs = [split_idxs]
    if split_idxs.ndim == 0:
        split_idxs = np.array([split_idxs])
    #print split_idxs
    try:
        idx_list = np.split(idxs,split_idxs)
    except ValueError:
        #print 'value error'
        np.split(idxs,split_idxs)
        return None
    idx_list = [x[1:] for x in idx_list]
    idx_list = [x for x in idx_list if len(x)>0]
    return idx_list

def rewrap(trace,offset = np.pi/2.):
    unwrapped = np.unwrap(trace,np.pi*1.8)
    vel = np.diff(unwrapped)
    return np.mod(unwrapped+np.deg2rad(offset),1.9*np.pi),vel

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    import numpy as np
    return np.isnan(y), lambda z: z.nonzero()[0]

def fill_nan(y): 
    import numpy as np
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y
        
def butter_bandpass(lowcut, highcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0/sampling_period
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, sampling_period, order=5):
    import scipy.signal
    b, a = butter_bandpass(lowcut, highcut, sampling_period, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_lowpass(lowcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0/sampling_period
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    b, a = scipy.signal.butter( order, low, btype='low')
    return b, a

def butter_lowpass_filter(data, lowcut, sampling_period, order=5):
    import scipy.signal
    b, a = butter_lowpass(lowcut, sampling_period, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_highpass(highcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0/sampling_period
    nyq = 0.5 * sampling_frequency
    high = highcut / nyq 
    b, a = scipy.signal.butter( order, high, btype='high')
    return b, a

def butter_highpass_filter(data, highcut, sampling_period, order=5):
    import scipy.signal
    b, a = butter_highpass(highcut, sampling_period, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y
    