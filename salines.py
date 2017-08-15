#!/Library/Frameworks/Python.framework/Versions/Current/bin/python

#  salines.py
#  
#
#  Created by Theodore Lindsay on 10/2/09.
#  Copyright (c) 2009 University of Oregon. All rights reserved.
#
"""calculate recipe for a volume of saline given the desired concentrations"""

import quantities as pq
#from psilentplib import Struct
pq.molar = pq.UnitQuantity('molar', pq.mol/pq.liter, symbol='M')

class Struct:
    """general structure class"""
    def __init__ (self, *argv, **argd):
        if len(argd):
            # Update by dictionary                                                                                            
            self.__dict__.update (argd)
        else:
            # Update by position                                                                                              
            attrs = filter (lambda x: x[0:2] != "__", dir(self))
            for n in range(len(argv)):
                setattr(self, attrs[n], argv[n])

class ion(object):
    def __init__(self,species,concentration):
        self.species = species 
        self.conc = concentration
        self.mob = mob(species)
        self.val = get_valence(species)
        
class solute(object):
    def __init__(self,**kwargs):
        concentration = kwargs.pop('concentration', pq.Quantity(1,pq.molar,dtype = 'Float32'))
        self.concentration = pq.Quantity(concentration,pq.molar)
        stock = kwargs.pop('stock', pq.Quantity(1,pq.molar,dtype = 'Float32'))
        self.stock = pq.Quantity(stock,pq.molar)
        hydration =  kwargs.pop('hydration',0)
        self.hydration = hydration
        FW = kwargs.pop('FW', pq.Quantity(1,pq.g/pq.mol,dtype = 'Float32'))
        self.FW = pq.Quantity(FW,pq.g/pq.mol)
        self.ions = kwargs.pop('ions')
                
    def mass_for_volume(self,volume):
        volume = pq.Quantity(volume,pq.liter)
        r_val = volume*self.concentration*self.FW + self.hydration*self.wm_for_volume(volume)
        r_val.units = 'g'
        return r_val
        
    def wm_for_volume(self,volume):
        r_val = volume*self.concentration*pq.Quantity(18.01528,pq.g/pq.mol,dtype = 'Float32')
        r_val.units = 'g'
        return r_val
        
    def stock_for_volume(self,volume):
        volume = pq.Quantity(volume,pq.liter)
        r_val = volume * self.concentration / self.stock
        r_val.units = 'mL'
        return r_val
        
    def str_for_volume(self,volume):
        r_str = ''
        stock_volume = self.stock_for_volume(volume)
        grams = self.mass_for_volume(volume)
        concentration = self.concentration
        stock = '(' + str(self.stock.magnitude) + 'M' +')'
        r_str = r_str + str(str(concentration.magnitude) + ' ' + concentration.dimensionality.string).rjust(14)
        r_str = r_str + str(str(stock_volume.magnitude) + ' ' + stock_volume.dimensionality.string).rjust(15)
        r_str = r_str + stock.rjust(8)
        r_str = r_str + str('%.4f'%grams.magnitude + ' ' + grams.dimensionality.string).rjust(15) + '\n'
        return r_str
        
    def ion_concentration(self,ion):
        return self.ions.count(ion)*self.concentration
        
class saline(dict):
    def __init__(self,**params):
        dict.__init__(self,**params)
        volume = self.pop('volume',pq.Quantity(1,pq.liter,dtype = 'Float32'))
        name = self.pop('name','')
        self.volume = pq.Quantity(volume,pq.liter)
        self.name = name
        
    def set_volume(self,*args):
        volume = args[0]
        if not (type(volume) is pq.Quantity):
            try:
                units = args[1]
            except IndexError:
                units = 'L'
            volume = pq.Quantity(volume,units,dtype = 'Float32')
        self.volume = volume
        
    def __str__(self):
        r_str = self.name + '\n' + 'For ' + str(self.volume.magnitude) + ' ' + self.volume.dimensionality.string
        r_str += ' of saline---:\n'
        r_str += 'Reagent' +'Concentration'.center(20) + 'Stock'.center(20) + 'Mass'.center(13) 
        r_str += '\n--------  -------------      -------------        --------\n'
        for k in self.keys():
            r_str = str(r_str + k + ':' + '\t')
            r_str = r_str + self[k].str_for_volume(self.volume)
        r_str += '\nion concentrations:\n-------------------------------------------\n'
        cons = [s + ':' + str(self.ion_concentration(s)) + ' ' for  s in self.get_ion_set()]
        for s in cons:
            r_str += s
            r_str += '\n'
        total = pq.Quantity(0,pq.molar)
        for ion in self.get_ion_set():
            total = total + self.ion_concentration(ion)
        r_str += 'calculated osmolality:' + str(total) + '\n'
        return r_str
        
    def html(self):
        r_str = '<strong>' + self.name + '</strong>' + '</br></br>' + 'For ' + str(self.volume.magnitude) + ' ' + self.volume.dimensionality.string
        r_str += ' of saline:</br>'
        r_str += '<table>'
        r_str += '<tr><th>Reagent</th><th>Concentration</th><th>Stock</th><th>Mass</th><tr>'
        for k in self.keys():
            stock_volume = self[k].stock_for_volume(self.volume)
            grams = self[k].mass_for_volume(self.volume)
            concentration = self[k].concentration
            stock = ' (' + str(self[k].stock.magnitude) + 'M' +')'
            
            r_str += '<tr>'
            r_str = str(r_str + '<td>' + k + '</td>')
            r_str += '<td>' + str(concentration.magnitude) + ' ' + concentration.dimensionality.string + '</td>'
            r_str += '<td>' + str(stock_volume.magnitude) + ' ' + stock_volume.dimensionality.string + stock +'</td>'
            r_str += '<td>' + '%.4f'%grams.magnitude + ' ' + grams.dimensionality.string + '</td>'
            r_str += '</tr>'
        
        r_str += '</table>'
        r_str += '</br><strong>ion concentrations:</strong></br>'
        cons = [s + ': ' + str(self.ion_concentration(s)) + ' ' for  s in self.get_ion_set()]
        for s in cons:
            r_str += s
            r_str += '</br>'
        total = pq.Quantity(0,pq.molar)
        for ion in self.get_ion_set():
            total = total + self.ion_concentration(ion)
        r_str += '<strong>calculated osmolality:</strong>' + str(total) + '</br>'
        return r_str
    
    
    def ion_concentration(self,ion):
        r_val = pq.Quantity(0.0,pq.molar)
        for x in self.keys():
            r_val += self[x].ion_concentration(ion)
        return r_val
    
    def get_ion_set(self):
        ion_set = set()
        for x in self.keys():
            ion_set.update(self[x].ions)
        ion_set = list(ion_set)
        ion_set.sort()
        return ion_set

class ext(Struct):
    class tets(Struct):
        mod2 = saline(name = 'ext.tets.mod2',volume = 0.200)
        mod2['NaCl'] = solute(concentration = 0.145,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        mod2['KCl'] = solute(concentration = 0.005,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        mod2['MgCl2'] = solute(concentration = 0.005,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        mod2['CaCl2'] = solute(concentration = 0.001,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        mod2['HEPES'] = solute(concentration = 0.010,stock = 1, FW = 238.30,ions = ['HEPES-','H+'])
        
        highCa = saline(name = 'ext.tets.highCa',volume = 0.200)
        highCa['NaCl'] = solute(concentration = 0.143,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        highCa['KCl'] = solute(concentration = 0.005,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        highCa['CaCl2'] = solute(concentration = 0.008,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        highCa['HEPES'] = solute(concentration = 0.010,stock = 1, FW = 238.30,ions = ['HEPES-','H+'])
        
    class mk801(Struct):
        mod1 = saline(name = 'ext.mk801.mod1',volume = 0.010)
        mod1['NaCl'] = solute(concentration = 0.145,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        mod1['KCl'] = solute(concentration = 0.005,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        mod1['MgCl2'] = solute(concentration = 0.005,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        mod1['CaCl2'] = solute(concentration = 0.001,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        mod1['HEPES'] = solute(concentration = 0.010,stock = 1, FW = 238.30,ions = ['HEPES-','H+'])
        mod1['MK801'] = solute(concentration = 0.000050,stock = 0.01, FW = 238.30,ions = ['MK801+','HM-'])
        
    class CNQX(Struct):
        mod1 = saline(name = 'ext.CNQX.mod1',volume = 0.01)
        mod1['NaCl'] = solute(concentration = 0.145,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        mod1['KCl'] = solute(concentration = 0.005,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        mod1['MgCl2'] = solute(concentration = 0.005,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        mod1['CaCl2'] = solute(concentration = 0.001,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        mod1['HEPES'] = solute(concentration = 0.010,stock = 1, FW = 238.30,ions = ['HEPES-','H+'])
        mod1['CNQX'] = solute(concentration = 0.0002,stock = 0.01, FW = 276.12,ions = ['CNQX','Na+','Na+'],hydration = 0)
        
class internal(Struct):
    class buffCa(Struct):
        mod2 = saline(name = 'internal.bufCa.mod2',volume = 0.050)
        mod2['Kgluc'] = solute(concentration = 0.125,stock = 1, FW = 234.25, ions = ['K+','gluc-'])
        mod2['NaCl'] = solute(concentration = 0.004,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        mod2['KCl'] = solute(concentration = 0.018,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        mod2['MgCl2'] = solute(concentration = 0.001,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        mod2['CaCl2'] = solute(concentration = 0.001,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        mod2['HEPES'] = solute(concentration = 0.010,stock =  1, FW = 238.30,ions = ['HEPES-','H+'])
        mod2['EGTA'] = solute(concentration = 0.010,stock = 1, FW = 380.35,ions = ['EGTA--','H+','H+'])

        cellsignals = saline(name = 'int.buffCa.cellsignals',volume = 0.002)
        cellsignals['Kgluc'] = solute(concentration = 0.075,stock = 1, FW = 234.25, ions = ['K+','gluc-'])
        cellsignals['NaCl'] = solute(concentration = 0.001,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        cellsignals['KCl'] = solute(concentration = 0.018,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        cellsignals['MgCl2'] = solute(concentration = 0.001,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        cellsignals['CaCl2'] = solute(concentration = 0.0006,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        
        cellsignals['KATP'] = solute(concentration = 0.005,stock = 0.01, FW = 619.39,ions = ['K+','K+','ATP--'])
        cellsignals['NaGTP'] = solute(concentration = 0.001,stock = 0.01, FW = 523.18,ions = ['Na+','Na+','GTP--'])
        cellsignals['cAMP'] = solute(concentration = 0.0005,stock = 0.01, FW = 369.20,ions = ['Na+','cAMP-'])
        cellsignals['cGMP'] = solute(concentration = 0.0005,stock = 0.01, FW = 367.19,ions = ['Na+','cGMP-'])
        
        cellsignals['HEPES'] = solute(concentration = 0.010,stock = 1, FW = 238.30,ions = ['HEPES-','H+'])
        cellsignals['BAPTA'] = solute(concentration = 0.005,stock = 1, FW = 628.8,ions = ['BAPTA','K+','K+','K+','K+'])
        cellsignals['EGTA'] = solute(concentration = 0.005,stock = 1, FW = 380.35,ions = ['EGTA--'])
        cellsignals['CrPO4'] = solute(concentration = 0.010,stock = 1, FW = 287.3,ions = ['Cre--','K+','K+'])
        cellsignals['Speram'] = solute(concentration = 0.010,stock = 1, FW = 202.34,ions = ['Spe'])
        
        lowCl = saline(name = 'internal.bufCa.lowCl',volume = 0.200)
        lowCl['Kgluc'] = solute(concentration = 0.143,stock = 1, FW = 234.25, ions = ['K+','gluc-'])
        lowCl['NaCl'] = solute(concentration = 0.004,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        lowCl['MgCl2'] = solute(concentration = 0.001,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        lowCl['CaCl2'] = solute(concentration = 0.001,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        lowCl['HEPES'] = solute(concentration = 0.010,stock =  1, FW = 238.30,ions = ['HEPES-','H+'])
        lowCl['EGTA'] = solute(concentration = 0.010,stock = 1, FW = 380.35,ions = ['EGTA--','H+','H+'])

    class perfpatch(Struct):
        amp = saline(name = 'int.perfpatch.amp',volume = 0.006)
        amp['Kgluc'] = solute(concentration = 0.125,stock = 1, FW = 234.25, ions = ['K+','gluc-'])
        amp['NaCl'] = solute(concentration = 0.004,stock = 2, FW = 58.44, ions = ['Na+','Cl-'])
        amp['KCl'] = solute(concentration = 0.018,stock = 1, FW = 74.55,ions = ['K+','Cl-'])
        amp['MgCl2'] = solute(concentration = 0.001,stock = 1, FW = 95.21,ions = ['Mg++','Cl-','Cl-'])
        amp['CaCl2'] = solute(concentration = 0.001,stock = 1, FW = 110.98,ions = ['Ca++','Cl-','Cl-'])
        amp['HEPES'] = solute(concentration = 0.010,stock =  1, FW = 238.30,ions = ['HEPES-','H+'])
        amp['EGTA'] = solute(concentration = 0.010,stock = 1, FW = 380.35,ions = ['EGTA--','H+','H+'])
        #Solubulized Amphotericin B, lot 118k4029. MW = 924.08 * 2.57 (lot potency correction factor)
        amp['AmpB'] = solute(concentration = 0.006,stock = 1, FW = 2374.88,ions = ['AmpB'])

def ljp(internal,external):
    internals = dict()
    externals = dict()
    for i in internal.get_ion_set():
        internals[i] = ion(i,internal.ion_concentration(i))
    for i in external.get_ion_set():
        externals[i] = ion(i,external.ion_concentration(i))
    for key in internals.keys():
        i = internals[key]
        num = i.val**2*i.mob*i.conc
        #return i.conc

def mob(ion):
    mob_table = {'H+':4.757,
                 'Rb+':1.059,
                 'Cs+':1.050,
                 'K+':1.000,
                 'Ag+':0.842,
                 'Na+':0.682,
                 'Li+':0.525,
                 'NH4+':1.000,
                 'Tl+':1.02,
                 'TMA+':0.611,
                 'TEA+':0.444,
                 'TprA':0.318,
                 'Ca++':0.4048,
                 'Mg++':0.361,
                 'Sr++':0.404,
                 'gluc-':0.33,
                 'Cl-':1.0388,
                 'EGTA--':0.24,
                 'HEPES-':0.30}
    return mob_table[ion]          
    
def saline_nersts(ext,int,temp = pq.Quantity(25,'degC')):
    if type(temp) != pq.Quantity:
        temp = pq.Quantity(temp,'degC')
    ions = ext.get_ion_set()
    nersts = dict()
    for ion in ions:
        try:
            cE = ext.ion_concentration(ion)
            cI = int.ion_concentration(ion)
        except(KeyError):
            print "no ion"
        E_ion = nerst(ion,cE,cI,temp=temp)
        E_ion.units = 'mV'
        nersts.update({str(ion):E_ion})
    return nersts
        #print ion + ':' + str(E_ion)

def nerst(ion,cE,cI,temp = pq.Quantity(25,'degC')):
    import numpy as np
    if type(temp) != pq.Quantity:
        temp = pq.Quantity(temp,'degC')
    if temp.dimensionality.keys()[0] is not pq.degK: #rescale to K if nessesary
        temp = temp + pq.Quantity(273.15,'degC')
        temp.units = pq.K
    #get valence
    z = 0
    if ion[-1] == '+':
        z += 1  
    if ion[-2] == '+':
        z += 1
    if ion [-1] == '-':
        z -= 1
    if ion [-2] == '-':
        z -= 1
    if z == 0:
        return None
    F = pq.constants.Faraday_constant
    R = pq.constants.R
    return R*temp/(z*F) * np.log(cE/cI)
    
def ghk_voltage_monovalent(ext,int,temp= pq.Quantity(25,'degC'),p_K = 1,p_Na = 1,p_Cl = 0): 
    import numpy as np
    if type(temp) != pq.Quantity:
        temp = pq.Quantity(temp,'degC')
    if temp.dimensionality.keys()[0] is not pq.degK: #rescale to K if nessesary
        temp = temp + pq.Quantity(273.15,'degC')
        temp.units = pq.K

    F = pq.constants.Faraday_constant
    R = pq.constants.R
    
    c_Na_e = ext.ion_concentration('Na+')
    c_Na_i = int.ion_concentration('Na+')
    c_K_e = ext.ion_concentration('K+')
    c_K_i = int.ion_concentration('K+')
    c_Cl_e = ext.ion_concentration('Cl-')
    c_Cl_i = int.ion_concentration('Cl-')
    rval = -1* R*temp/F * np.log((p_Na*c_Na_i + p_K*c_K_i + p_Cl*c_Cl_e)/(p_Na*c_Na_e +p_K*c_K_e + p_Cl*c_Cl_i))
    rval.units = 'mV'
    return rval
    
def ghk_voltage(ext,int,temp = pq.Quantity(25,'degC'),p_K = 1,p_Na = 1,p_Cl = 0,p_Ca = 0,p_Mg = 0):
    ifunc = lambda n: ghk_current(ext,int,n,p_K = p_K,p_Na = p_Na,p_Cl = p_Cl,p_Ca = p_Ca,p_Mg = p_Mg)
    import scipy.optimize as optm
    return optm.fsolve(ifunc,-100)
    
    #return ifunc
    
def ghk_current(ext,int,V,temp = pq.Quantity(25,'degC'),p_K = 1,p_Na = 1,p_Cl = 0,p_Ca = 0,p_Mg = 0):
    import numpy 
    net_current = pq.Quantity(0,'A/m**2')
    ions = ext.get_ion_set()
    for ion,perm in zip(['K+','Na+','Cl-','Ca++','Mg++'],[p_K,p_Na,p_Cl,p_Ca,p_Mg]):
        try:
            cE = ext.ion_concentration(ion)
            cI = int.ion_concentration(ion)
            cE = cE.rescale('mol/L')
            cI = cI.rescale('mol/L') 
        except(KeyError):
            print "no ion"
        #print ion,cE,cI,perm,V,temp
        #print ghk_current_ion(ion,cE,cI,P_ion = perm,V = V,temp = temp)[0]
        temporary = ghk_current_ion(ion,cE,cI,P_ion = perm,V = V,temp = temp)
        if numpy.ndim(temporary)>0:
            temporary = temporary[0]
        net_current += temporary
    return net_current

    
def ghk_current_ion(ion,cE,cI,P_ion = 1,V = pq.Quantity(1,'mV'),temp = pq.Quantity(25,'degC')):
    import numpy as np
    if type(temp) != pq.Quantity:
        temp = pq.Quantity(temp,'degC')
    if temp.dimensionality.keys()[0] is not pq.degK: #rescale to K if nessesary
        temp = temp + pq.Quantity(273.15,'degC')
        temp.units = pq.K
    if V == 0: #GHK equation is undefined at V = 0 
        V = 1e-10
    V = pq.Quantity(V,'mV')
    #units of permeability are meters/sec
    P_ion = pq.Quantity(P_ion,'m/sec')
    #get valence
    z = 0
    if ion[-1] == '+':
        z += 1
    if ion[-2] == '+':
        z += 1
    if ion [-1] == '-':
        z -= 1
    if ion [-2] == '-':
        z -= 1
    if z == 0:
        return None
    F = pq.constants.Faraday_constant
    R = pq.constants.R
    RToF = (R*temp)/F
    RToF.units = 'mV'
    eterm = -1 * z * V * 1/RToF
    i = ((cI - cE * np.exp(eterm))/(1-np.exp(eterm)))
    pterm = P_ion * z**2 * F * eterm
    i = pterm * i
    i.units = 'A/m**2'
    return i

def get_valence(ion):
    z = 0
    if ion[-1] == '+':
        z += 1
    if ion[-2] == '+':
        z += 1
    if ion [-1] == '-':
        z -= 1
    if ion [-2] == '-':
        z -= 1
    if z == 0:
        print ion
        return None
    else:
        return z
    
def test():
    a = internal.buffCa.mod2
    print(str(a))
    b = internal.buffCa.cellsignals
    print(str(b))
    
if __name__ == '__main__':
    test()
        
