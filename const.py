import healpy as hp
import numpy as np
from scipy import stats

c = 3e10
k = 1.381e-16
pc = 3.086e18
d = 8e3*pc # distance from galactic center to sun ~8kpc
NSIDE_sky = 256
NSIDE_dg = 32
NPIX_dg = hp.nside2npix(NSIDE_dg)
m = np.arange(NPIX_dg)
coords = hp.pix2ang(NSIDE_dg, m, lonlat=True)
l = coords[0]
b = coords[1]
nu_ref = 1420e6
T_eg = 0.08866 # K
a = -0.7 # spectral index
T_CMB = 2.725 # K
b_mask = 10. # deg
idx_exb = hp.query_strip(NSIDE_dg, np.deg2rad(90-b_mask), np.deg2rad(90+b_mask))


