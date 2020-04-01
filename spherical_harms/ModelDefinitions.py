# function describing line of sight through disk and halo models, based on sky coordinates, 
# geometry of disk/halo
# l and b given in degrees (as from hp.pix2ang) - convert to radians
# d = distance of sun from center of galaxy
# R_disk, h_disk, R_halo can all be arrays

import numpy as np
import healpy as hp
from const import *
import matplotlib.pyplot as plt

def LineOfSightDisk(ll, bb, dd, R_disk, h_disk):
    
    # first calculate length of line of sight through disk
    
    l1 = np.array(ll)
    b1 = np.array(bb)
    
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)

    l1 = np.array([np.min([l_, (2*np.pi)-l_]) for l_ in l1])
    b1 = np.abs(b1)
    B_disk =  -l1 - np.arcsin((dd/R_disk)*np.sin(l1)) + np.pi
    r_disk = np.sqrt(-(2.*R_disk*dd*np.cos(B_disk)) + (R_disk**2) + (dd**2))
    
    b_crit = np.arctan(0.5*h_disk/r_disk)
    b_below = np.nan_to_num((b1 <= b_crit)*r_disk/(np.cos(b1)))
    b_above = np.nan_to_num((b1 > b_crit)*0.5*h_disk/(np.sin(b1)))

    D_disk = b_below+b_above
        
    return np.array(D_disk)
        

def LineOfSightHalo(ll, bb, dd, R_halo):  
    
    # calculate length of line of sight through halo
    
    l1 = np.deg2rad(np.array(ll))
    b1 = np.deg2rad(np.array(bb))
    
    l1 = np.minimum(l1, (2*np.pi)-l1)
    
    b1 = np.abs(b1)
    b_ = np.pi - b1
    
    d_proj = dd*np.abs(np.cos(l1))
    B_halo1 = np.pi - l1 - np.arcsin((dd/R_halo)*np.sin(l1))
    
    R_eff_above = np.sqrt((R_halo**2) + (dd**2) - (2*R_halo*dd*np.cos(B_halo1))) + d_proj
    B_halo_above = (np.pi - b_ - np.arcsin((d_proj/R_eff_above)*np.sin(b1)))
    D_tot_above = (l1 >= np.pi/2)*(np.sqrt((R_eff_above**2) + (d_proj**2) - (2*R_eff_above*d_proj*np.cos(B_halo_above))))
        
    R_eff_below = np.sqrt((R_halo**2) + (dd**2) - (2*R_halo*dd*np.cos(B_halo1))) - d_proj
    B_halo_below = (np.pi - b1 - np.arcsin((d_proj/R_eff_below)*np.sin(b1)))
    D_tot_below = (l1 < np.pi/2)*(np.sqrt((R_eff_below**2) + (d_proj**2) - (2*R_eff_below*d_proj*np.cos(B_halo_below))))

    # fixes bug for low R_halo (<~2.3 d)
    D_tot_above[np.isnan(D_tot_above)] = 0.
    D_tot_below[np.isnan(D_tot_below)] = 0.
    
    D_halo = D_tot_above + D_tot_below
    
    return D_halo

# Construct a T_b vs csc(b) graph
def cscbplot(Tmap, NSIDE):

    b_range = np.linspace(10,90,15)

    cscb = []
    Tb_mean = []

    for i in range(len(b_range)-1):

        b1 = 90 - b_range[i]
        b2 = 90 - b_range[i+1]

        bmid = 0.5*(b_range[i] + b_range[1+i])

        cscb.append(1/(np.sin(np.deg2rad(bmid)))) 

        idx = hp.query_strip(NSIDE, np.deg2rad(b2), np.deg2rad(b1))
        Tb_mean.append(np.mean(Tmap[idx]))
        
    return [cscb, Tb_mean]

# return Alm and sig for map
def prep_data(Tmap):

    # lower resolution
    map_dg = hp.pixelfunc.ud_grade(Tmap, NSIDE_dg)

    # do transform
    Alm = ((hp.sphtfunc.map2alm(map_dg,pol=False,gal_cut=b_mask)).real)
    Cl = hp.sphtfunc.anafast(map_dg,alm=False,pol=False,gal_cut=b_mask)

    # generate sigmas
    sig2 = np.zeros(len(Alm))
    lma = hp.sphtfunc.Alm.getlmax(len(Alm))
    l,m = hp.sphtfunc.Alm.getlm(lma)
    
    for i in range(len(Alm)):
        
        if l[i]<2:
            sig2[i] = Cl[2]
        else:
            sig2[i] = Cl[l[i]]

    return Alm[lm_idx],sig2[lm_idx]
        

# function to simulate a map. To just simulate a disk / halo, set j_halo / j_disk to 0. 
def sim_map(T_bkg, R_disk, h_disk, j_disk, R_halo, j_halo, noise = 0.01):

    model = LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2)) + T_bkg + LineOfSightHalo(l, b, d, R_halo)*j_halo*(c**2)/(2*k*(nu**2))

    model += noise*np.random.normal(size=len(model))

    return model


