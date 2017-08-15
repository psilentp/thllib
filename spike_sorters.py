import numpy as np
import neo

class SpikePool(neo.SpikeTrain):
    """hold a Sub group of spikes for further processing - keep track of spike
    pool and spike pool indecies so it is easy to mix the results back in"""
    def __init__(self,*args,**argv):
        super(SpikePool,self).__init__(args,argv)
        self.wv_mtrx = np.vstack([np.array(wv) for wv in self.waveforms])
        self.time_mtrx = np.vstack([wf.times for wf in self.waveforms])
        self.spk_ind = np.arange(0,np.shape(self.wv_mtrx)[0])
            
    def copy_slice(self,sli,rezero = False):
        """copy a spiketrain with metadata also enable slicing"""
        st = self[sli]
        if rezero:
            shift = st.waveforms[0].times[0]
        else:
            shift = pq.Quantity(0,'s')
        wvfrms = [neo.AnalogSignal(np.array(wf),
                               units = wf.units,
                               sampling_rate = wf.sampling_rate,
                               name = wf.name,
                               t_start = wf.t_start - shift)
                               for wf in st.waveforms]
        #pk_ind = self.annotations['pk_ind'][sli]
        t_start = wvfrms[0].times[0]
        t_stop = wvfrms[-1].times[-1]
        return SpikePool(np.array(st)-float(shift),
                        units = st.units,
                        sampling_rate = st.sampling_rate,
                        waveforms = wvfrms,
                        left_sweep = st.left_sweep,
                        t_start = t_start,
                        t_stop = t_stop)
        
    def __reduce__(self):
        return _new_spikepool, (self.__class__, np.array(self),
                                 self.t_stop, self.units, self.dtype, True,
                                 self.sampling_rate, self.t_start,
                                 self.waveforms, self.left_sweep,
                                 self.name, self.file_origin, self.description,
                                 self.annotations)
    
class SpkCollection(object):
    """class to hold the data for a collection of spikes.holds the matrx of waveforms
    and indxs from the trace that correspond to those waveforms""" 
    def __init__(self,spike_pool,selection_mask,params):
        self.spike_pool = spike_pool
        self.selection_mask = selection_mask
        self.params = params
        
    def collection_wvmtrx(self):
        """return the matrix of waveforms"""
        return self.spike_pool.wv_mtrx[self.collection_ind(),:]
        
    def collection_ind(self):
        """return the indices of the spikes"""
        return self.spike_pool.spk_ind[np.argwhere(self.selection_mask)[:,0]]
        
class SpkSelector(SpkCollection):
    """object that will mask out a matrix of spikes from a list of labels"""
    def __init__(self,spike_pool,selection_mask,input_mtrx,params):
        """init a collection of spikes and associated labels - for instance, different
        clusters from a k means....""" 
        super(SpkSelector,self).__init__(spike_pool,selection_mask,params)
        self.input_mtrx = input_mtrx
        self.labels = np.zeros(np.shape(selection_mask),dtype = 'S20')
        
    def mask_from_labels(self,select_labels):
        """return a mask of everyting masked out except the labels in select labels"""
        select_labels = np.array(select_labels)
        mask = np.in1d(self.labels,select_labels)
        return mask
        
    def ind_from_labels(self,select_labels):
        """convenience function to return the indices of select labels"""
        mask = self.mask_from_labels(select_labels)
        return self.spike_pool.spk_ind[np.argwhere(mask)[:,0]]
        
    def wv_mtrx_from_labels(self,select_labels):
        """return a matrix of waveforms from just the selected lables"""
        return self.spike_pool.wv_mtrx[self.ind_from_labels(select_labels)]
        
    def select(self):
        """abstract method to generate the labels - inheriting
        classes should set self.labels[self.collection_ind()] to something"""
        pass

class SpkTransformer(SpkCollection):
    """spike collection that can return a matrix of transformed waveforms for further 
    sorting"""
    def __init__(self,spike_pool,selection_mask,params):
        super(SpkTransformer,self).__init__(spike_pool,selection_mask,params)
        self.trnsmtrx = np.zeros((np.shape(self.spike_pool.wv_mtrx)[0],
                                 self.params['trans_dims']))
        
    def collection_trnsmtrx(self):
        """return the transformed matrxi"""
        return self.trnsmtrx[self.collection_ind(),:]
    
    def transform(self):
        """perform the transformation inheriting classes should set self.trnsmtrx.
        can assume to have a subsample in self.collection_ind() to improve efficiency"""
        pass

class SampleRandomSeq(SpkSelector):
    def select(self):
        seq_len = self.params['seq_len']
        n_seq = self.params['n_seq']
        import random
        idx = self.collection_ind()
        self.seq_starts = random.sample(idx[::seq_len],n_seq)
        self.seq_starts.sort()
        for i,st in enumerate(self.seq_starts):
            self.labels[st:st+seq_len] = 'seq%s'%(i)

class SpectralCluster(SpkSelector):
    def select(self):
        from sklearn.metrics import euclidean_distances
        wv_mtrx = self.collection_wvmtrx()
        #print "computing distances"
        #distances = euclidean_distances(wv_mtrx,wv_mtrx)
        from sklearn.cluster import SpectralClustering
        self.est = SpectralClustering(n_clusters=2,
                                      affinity="nearest_neighbors")
        #print "fitting"
        self.est.fit(wv_mtrx)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels
        
class P2PTransform(SpkTransformer):
    def transform(self):
        wv_mtrx = self.collection_wvmtrx()
        p2p = np.max(wv_mtrx,axis = 1) -np.min(wv_mtrx,axis = 1)
        p2pt = np.argmax(wv_mtrx,axis = 1) - np.argmin(wv_mtrx,axis = 1)
        print(np.shape(p2p))
        print(np.shape(p2pt))
        self.trnsmtrx = np.hstack((np.array([p2p]).T,np.array([p2pt]).T))
    
    
class KMeansCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import KMeans
        X = self.input_mtrx[self.collection_ind()]
        self.est = KMeans(n_clusters= self.params['kmeans_nc'],
                          init = self.params['init'])
        self.est.fit(X)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels
        
class PCATransform(SpkTransformer):
    def transform(self):
        from sklearn import decomposition
        wv_mtrx = self.collection_wvmtrx()
        self.est = decomposition.PCA(n_components=self.params['trans_dims'],
                                     whiten = self.params['pca_whiten'])
        self.est.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = self.est.transform(wv_mtrx)
        

class DBSCANCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        X = self.input_mtrx[self.collection_ind()]
        #X = StandardScaler().fit_transform(X)
        self.est = DBSCAN(eps = self.params['eps'],
                          min_samples = self.params['min_samples'])
        self.est.fit(X)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels.astype(int)

class WTR(SpkTransformer):
    def __init__(self,spike_pool,selection_mask,params):        
        super(SpkTransformer,self).__init__(spike_pool,selection_mask,params)
        #self.trnsmtrx = np.zeros((np.shape(self.spike_pool.wv_mtrx)[0],self.params['trans_dims']))
        
    def transform(self):
        import pywt
        X = self.collection_wvmtrx()
        wavelet = pywt.Wavelet(self.params['wavelet'])
        
        n_samples = X.shape[1]
        n_spikes = X.shape[0]
        
        def full_coeff_len(datalen, filtlen, mode):
            max_level = pywt.dwt_max_level(datalen, filtlen)
            total_len = 0
            for i in xrange(max_level):
                datalen = pywt.dwt_coeff_len(datalen, filtlen, mode)
                total_len += datalen 
            return total_len + datalen
        
        n_features = full_coeff_len(n_samples, wavelet.dec_len, 'sym')
        
        est_mtrx = np.zeros((n_spikes,n_features))
        self.trnsmtrx = np.zeros((np.shape(self.spike_pool.wv_mtrx)[0],n_features))
        for i in xrange(n_spikes):
            tmp = np.hstack(pywt.wavedec(X[i, :], wavelet, 'sym'))
            est_mtrx[i, :] = tmp
        self.trnsmtrx[self.collection_ind(),:] = est_mtrx

class WvltPCA(SpkTransformer):
    def transform(self):
        wvlt_params = {'wavelet':'db1'}
        wt = WTR(self.spike_pool,self.selection_mask,wvlt_params)
        wt.transform()
        from sklearn import decomposition
        wv_mtrx = wt.trnsmtrx[self.collection_ind(),:]
        self.est = decomposition.PCA(n_components=self.params['trans_dims'],
                                     whiten = self.params['pca_whiten'])
        self.est.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = self.est.transform(wv_mtrx)
        
class MedTrans(SpkTransformer):
    def transform(self,inputmtrx = None):
        if not(inputmtrx == None):
            wv_mtrx = inputmtrx[self.collection_ind(),:]
        else:
            wv_mtrx = self.collection_wvmtrx()
        wv_mtrx = self.collection_wvmtrx()
        self.wv_med = np.median(wv_mtrx,axis = 0)
        self.resmtrx = wv_mtrx-self.wv_med
        err_vec = np.sum(np.sqrt(np.square(self.resmtrx)),axis = 1)
        #err_vec /= np.max(err_vec)
        self.trnsmtrx[self.collection_ind(),:] = err_vec[:,np.newaxis]

class ThreshSelector(SpkSelector):
    def select(self,thresh = 0.5):
        X = self.input_mtrx[self.collection_ind()]
        self.labels[self.collection_ind()] = np.array(X > thresh,dtype = int)
        
class MBKMeansCluster(SpkSelector):
    def select(self):
        from sklearn.cluster import MiniBatchKMeans
        X = self.input_mtrx[self.collection_ind()]
        self.est = MiniBatchKMeans(n_clusters= self.params['kmeans_nc'],
                          init = self.params['init'],batch_size = 20)
        self.est.fit(X)
        labels = self.est.labels_
        self.labels[self.collection_ind()] = labels
        
class GMMCluster(SpkSelector):
    def select(self):
        from sklearn import mixture
        X = self.input_mtrx[self.collection_ind()]
        self.est = mixture.DPGMM(n_components=3)
        self.est.fit(X)
        labels = self.est.predict(X)
        self.labels[self.collection_ind()] = labels

class PCATransform2(SpkTransformer):
    def transform(self,inputmtrx = None):
        from sklearn import decomposition
        if not(inputmtrx == None):
            wv_mtrx = inputmtrx[self.collection_ind(),:]
        else:
            wv_mtrx = self.collection_wvmtrx()
        self.est = decomposition.PCA(n_components=self.params['trans_dims'],
                                     whiten = self.params['pca_whiten'])
        self.est.fit(wv_mtrx)
        self.trnsmtrx[self.collection_ind(),:] = self.est.transform(wv_mtrx)

def plot_clusters(selector,plot_slice = slice(0,10,1)):
    import pylab as plb
    plb.figure(figsize=(2,4))
    mask = selector.selection_mask
    sp = selector.spike_pool
    times = np.array(sp.waveforms[0].times - sp.waveforms[0].times[0])
    peak_times = np.array(sp.times)[mask] - np.array([wf.times[0] for wf in sp.waveforms])[mask]
    wv_mtrx = sp.wv_mtrx[mask,:]
    labels = selector.labels[mask]
    for wf,lb,tm in zip(wv_mtrx[plot_slice],labels[plot_slice],peak_times[plot_slice]):
        try:
            plb.subplot(2,1,int(lb)+1)
            color_lookup = {'':'b','0':'r','1':'g'}
            color = color_lookup[lb]
            plb.plot(times,wf,color = color,alpha = 0.2)
            plb.plot(tm,1,'o')
        except ValueError:
            pass  
                                
def get_spike_pool(sweep,thresh = 10,wl=25,wr=20,filter_window = 35):
    from scipy.signal import medfilt
    detrend = np.array(sweep)-medfilt(sweep,filter_window)
    deltas = np.diff(np.array(detrend>thresh,dtype = 'float'))
    starts = np.argwhere(deltas>0.5)
    stops = np.argwhere(deltas<-0.5)
    if starts[0] > stops[0]:
        stops = stops[1:]
    if stops[-1] < starts[-1]:
        starts = starts[:-1]
    intervals = np.hstack((starts,stops))
    peaks = [np.argmax(sweep[sta:stp])+sta for sta,stp in intervals][2:-2]
    waveforms = [sweep[pk-wl:pk+wr] for pk in peaks]
    sweep.sampling_period.units = 's'
    pk_tms = sweep.times[np.array(peaks)]
    spike_pool = SpikePool(pk_tms,
                                sweep.t_stop,
                                sampling_rate = sweep.sampling_rate,
                                waveforms = waveforms,
                                left_sweep = wl*sweep.sampling_period,
                                t_start = sweep.t_start,
                                pk_ind = peaks)
    return spike_pool        