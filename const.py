import healpy as hp
import numpy as np
from scipy import stats

c = 3e10
k = 1.381e-16
pc = 3.086e18
d = 8e3*pc # distance from galactic center to sun ~8kpc
Jy = 1e-23
T_CMB = 2.725 # K


############ Reich and Reich all sky L-band map ###############

#nu_ref = 1420e6
#T_eg = 0.0887 # K (at 1.4 GHz)
#a = -0.7 # spectral index
#b_mask = 10. # deg

#NSIDE_sky = 256
#NSIDE_dg = 32
#NPIX_dg = hp.nside2npix(NSIDE_dg)
#idx_exb = hp.query_strip(NSIDE_dg, np.deg2rad(90-b_mask), np.deg2rad(90+b_mask))


#map_1420 = (hp.read_map('STOCKERT+VILLA-ELISA_1420MHz_1_256.fits'))/1000
#map_1420_dg = hp.pixelfunc.ud_grade(map_1420, NSIDE_dg)
#map_1420_dg[idx_exb] = None

#m = np.arange(NPIX_dg)
#coords = hp.pix2ang(NSIDE_dg, m, lonlat=True)
#l = coords[0]
#b = coords[1]



