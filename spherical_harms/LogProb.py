'''
Define log of likelihood function. 

Make sure all sky maps are in Kelvin 

'''

import ModelDefinitions as MD
import numpy as np
from scipy import stats
from const import *
import healpy as hp

# calculates Alm series for a model
def model(x,p0,p1,p2,p3,p4,p5):

    # get parameters
    T_bkg = p0
    R_disk = p1
    h_disk = p2
    j_disk = p3
    R_halo = p4
    j_halo = p5

    model = MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2)) + MD.LineOfSightHalo(l, b, d, R_halo)*j_halo*(c**2)/(2*k*(nu**2))#+ T_bkg
    Alm_model = hp.sphtfunc.map2alm(model,pol=False,gal_cut=b_mask)

    return (Alm_model.real)[lm_idx]

    

# ln(likelihood) for model with disk and bkg
def lik_disk(param, data):

    # get parameters
    T_bkg = param[0]
    R_disk = param[1]
    h_disk = param[2]
    j_disk = param[3]

    # calculate model
    model = MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2)) + T_bkg 
    Alm_model = ((hp.sphtfunc.map2alm(model,pol=False,gal_cut=b_mask)).real)[lm_idx]

    # get data and noise
    Alm,sig2 = data

    # get likelihood
    lik = np.sum((Alm_model-Alm)**2. / (sig2))

    return -lik
    
    
# ln(likelihood) for model with disk, bkg, and halo
def lik_disk_halo(param, data):

    # get parameters
    T_bkg = param[0]
    R_disk = param[1]
    h_disk = param[2]
    j_disk = param[3]
    R_halo = param[4]
    j_halo = param[5]

    # calculate model
    model = MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2)) + T_bkg + MD.LineOfSightHalo(l, b, d, R_halo)*j_halo*(c**2)/(2*k*(nu**2))
    Alm_model = ((hp.sphtfunc.map2alm(model,pol=False,gal_cut=b_mask)).real)[lm_idx]

    # get data and noise
    Alm,sig2 = data

    # get likelihood
    lik = np.sum((Alm_model-Alm)**2. / (sig2))

    return -lik
