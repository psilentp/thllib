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