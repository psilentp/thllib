import h5py
import os
import numpy as np
class NetFly(object):
    def __init__(self,flynum,rootpath = '/media/imager/FlyDataD/FlyDB/'):
        self.flynum = flynum
        self.flypath = rootpath + '/Fly%3d'%flynum
        self.flatpaths = [os.path.join(dp, f) for dp, dn, fn in os.walk(self.flypath) for f in fn]
        self.h5paths = [fn for fn in self.flatpaths if fn[-4:] == 'hdf5']
        self.txtpaths = [fn for fn in self.flatpaths if fn[-3:] == 'txt']
        self.pklpaths = [fn for fn in self.flatpaths if fn[-4:] == 'cpkl']
        self.script_name = [x for x in os.listdir(self.flypath) if '.py' in x][0].split('.py')[0]
        
    def open_signals(self,extensions = ['hdf5','txt','cpkl'],verbose = False):
        self.h5files = dict()
        if not(type(extensions) is list):
            extensions = [extensions]
        if 'hdf5' in extensions:
            if verbose: print('opening hdf5')
            for fn in self.h5paths:
                key = fn.split('/')[-1].split('.')[0]
                self.h5files[key] = h5py.File(fn,'r')
                self.__dict__[key] = self.h5files[key][key]
        if 'txt' in extensions:
            if verbose: print('opening txt')
            for fn in self.txtpaths:
                key = fn.split('/')[-1].split('.')[0]
                with open(fn,'rt') as f:
                    self.__dict__[key] = f.readlines()
        if 'cpkl' in extensions:
            import cPickle
            if verbose: print('opening pkl')
            for fn in self.pklpaths:
                key = fn.split('/')[-1].split('.')[0]
                with open(fn,'rb') as f:
                    self.__dict__[key] = cPickle.load(f)
    
    def save_pickle(self,data,filename):
        import cPickle
        if filename.split('.')[-1] == 'cpkl':
            with open(os.path.join(self.flypath,filename),'wb') as f:
                cPickle.dump(data,f)
            self.__dict__[filename.split('.')[0]] = data
        else:
            with open(os.path.join(self.flypath,filename + '.cpkl'),'wb') as f:
                cPickle.dump(data,f)
            self.__dict__[filename] = data
                
    def close_signals(self):
        for f in self.h5files.values():
            f.close()