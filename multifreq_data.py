import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.io import fits

import const as const
import TRIS_vals as tris

from const import *
from TRIS_vals import *


import importlib
importlib.reload(const);
importlib.reload(tris);

TRIS_bw = 2*11.48
l = TRIS_l[0.6e9]
b = TRIS_b[0.6e9]

###### 1.4 GHz Map #######

map1420 = hp.read_map('STOCKERT+VILLA-ELISA_1420MHz_1_256.fits')/1000
NSIDE_1420 = hp.get_nside(map1420)

map1420_bw = 35.4/60 #deg
map1420_smooth = hp.sphtfunc.smoothing(map1420, fwhm=np.deg2rad(np.sqrt(TRIS_bw**2 - map1420_bw**2)))
map1420_idx = hp.ang2pix(NSIDE_1420, l, b, lonlat=True)

map1420_data = map1420_smooth[map1420_idx] - T_CMB
map1420_errs = 0.5 #K 


###### TRIS 820 MHz #####

TRIS820_data = TRIS_Tb[0.82e9] - T_CMB
TRIS820_errs = TRIS_Tberrs[0.82e9]

###### TRIS 600 MHz #####

TRIS600_data = TRIS_Tb[0.6e9] - T_CMB
TRIS600_errs = TRIS_Tberrs[0.6e9]

###### 408 MHz Map #######

haslam = hp.read_map('haslam408_ds_Remazeilles2014.fits') #destriped only
NSIDE_haslam = hp.get_nside(haslam)

haslam_bw = 56/60 #deg
haslam_smooth = hp.sphtfunc.smoothing(haslam, fwhm=np.deg2rad(np.sqrt(TRIS_bw**2 - haslam_bw**2)))
haslam_idx = hp.ang2pix(NSIDE_haslam, l, b, lonlat=True)

haslam408_data = haslam_smooth[haslam_idx] - T_CMB
haslam408_errs = 3 # K

###### 150 MHz Map #########

map150 = hp.read_map('lambda_landecker_wielebinski_150MHz_SARAS_recalibrated_hpx_r8.fits')
NSIDE_map150 = hp.get_nside(map150)

map150_bw = 5 #deg
map150_smooth = hp.sphtfunc.smoothing(map150, fwhm=np.deg2rad(np.sqrt(TRIS_bw**2 - map150_bw**2)))
map150_idx = hp.ang2pix(NSIDE_map150, l, b, lonlat=True)

map150_data = map150_smooth[map150_idx] - T_CMB
map150_errs = 40 # K

########## All multi-freq data ##############
data = np.array([map1420_data, TRIS820_data, TRIS600_data, haslam408_data, map150_data])
errs = np.array([map1420_errs, TRIS820_errs, TRIS600_errs, haslam408_errs, map150_errs])
freqs = np.array([1420, 820, 600, 408, 150])*1e6





