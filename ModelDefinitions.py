# function describing line of sight through disk and halo models, based on sky coordinates, 
# geometry of disk/halo
# l and b given in degrees (as from hp.pix2ang) - convert to radians
# d = distance of sun from center of galaxy
# R_disk, h_disk, R_halo can all be arrays

import numpy as np

def LineOfSightDisk(l, b, d, R_disk, h_disk):
    
    # first calculate length of line of sight through disk
    
    l = np.array(l)
    b = np.array(b)
    
    l = np.deg2rad(l)
    b = np.deg2rad(b)

    l = np.array([np.min([l_, (2*np.pi)-l_]) for l_ in l])
    b = np.abs(b)
    B_disk =  -l - np.arcsin((d/R_disk)*np.sin(l)) + np.pi
    r_disk = np.sqrt(-(2*R_disk*d*np.cos(B_disk)) + (R_disk**2) + (d**2))
    
    b_crit = np.arctan(0.5*h_disk/r_disk)
    b_below = np.nan_to_num((b <= b_crit)*r_disk/(np.cos(b)))
    b_above = np.nan_to_num((b > b_crit)*0.5*h_disk/(np.sin(b)))
    
    D_disk = b_below+b_above
        
    return np.array(D_disk)
        

def LineOfSightHalo(l, b, d, R_halo):  
    
    # calculate length of line of sight through halo
    
    l = np.deg2rad(np.array(l))
    b = np.deg2rad(np.array(b))
    
    l = np.minimum(l, (2*np.pi)-l)
    
    b = np.abs(b)
    b_ = np.pi - b
    
    d_proj = d*np.abs(np.cos(l))
    B_halo1 = np.pi - l - np.arcsin((d/R_halo)*np.sin(l))
    
    R_eff_above = np.sqrt((R_halo**2) + (d**2) - (2*R_halo*d*np.cos(B_halo1))) + d_proj
    B_halo_above = (np.pi - b_ - np.arcsin((d_proj/R_eff_above)*np.sin(b)))
    D_tot_above = (l >= np.pi/2)*(np.sqrt((R_eff_above**2) + (d_proj**2) - (2*R_eff_above*d_proj*np.cos(B_halo_above))))
        
    R_eff_below = np.sqrt((R_halo**2) + (d**2) - (2*R_halo*d*np.cos(B_halo1))) - d_proj
    B_halo_below = (np.pi - b - np.arcsin((d_proj/R_eff_below)*np.sin(b)))
    D_tot_below = (l < np.pi/2)*(np.sqrt((R_eff_below**2) + (d_proj**2) - (2*R_eff_below*d_proj*np.cos(B_halo_below))))
    
    
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
