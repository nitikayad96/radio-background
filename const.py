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
nu = 1420e6
T_eg = 0.08866 # K
T_CMB = 2.725 # K
b_mask = 5. # deg
idx_exb = hp.query_strip(NSIDE_dg, np.deg2rad(90-b_mask), np.deg2rad(90+b_mask))


# PDF for ks test statistic sqrt(n)*D
LOC=0
SCALE=1
nD = np.linspace(stats.kstwobign.ppf(0.01, loc=LOC, scale=SCALE), stats.kstwobign.ppf(0.99, loc=LOC, scale=SCALE), 1000)
PDF = stats.kstwobign.pdf(nD, loc=LOC, scale=SCALE)
