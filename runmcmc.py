import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import emcee
from scipy import stats
import corner

import ModelDefinitions as MD
import LogProb as LP

### Set up known parameters ###
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

# make sure maps are in Kelvin
map_1420 = (hp.read_map('STOCKERT+VILLA-ELISA_1420MHz_1_256.fits'))/1000
map_1420_dg = hp.pixelfunc.ud_grade(map_1420, NSIDE_dg)
idx_exb = hp.query_strip(NSIDE_dg, np.deg2rad(90-10), np.deg2rad(90+10))

### set up priors for each MCMC run ###
# You change: value of priors #
def lnprior(param):
    
    R_disk = param[0]
    h_disk = param[1]
    j_disk = param[2]
    R_halo = param[3]
    j_halo = param[4]
    T_bkg = param[5]
    
    if d < R_disk < 100*d and 0 < h_disk < 5*d and 1e-42 < j_disk < 1e-39 and R_halo > 0 and 1e-43 < j_halo < 1e-38 and T_bkg > 0:
        return 0.0
    
    return -np.inf


### full log probability ###
# You change: likelihood function, chosen from LogProb #

def lnprob(param, nu, l, b, T_sky, T_eg, idx_exb):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + LP.diskhalobkg(param, nu, l, b, T_sky, T_eg, idx_exb)

### initialize walkers ###
# You change: ndim, nwalkers, param_init values
ndim = 6
nwalkers = 20
param_init = [2*d, 0.17*d, 2e-41, 4*d, 2.2e-42, 0.1]

init = [param_init]
for i in range(nwalkers-1):
    #start walkers such that all values are positive and mean multiplicative factor is 1
    vary = np.abs(np.random.randn(ndim)+1)
    init.append([param_init[i]*vary[i] for i in range(ndim)])
    
init = np.array(init)

### set up sampler and run MCMC ###
# You change: nsteps, filename
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(nu, l, b, map_1420_dg, T_eg, idx_exb), threads=4)

SAMPLER = sampler.run_mcmc(init, 100, progress=True)

samples_ = np.array(sampler.chain)
print('Average acceptance fraction: ', np.mean(sampler.acceptance_fraction))

np.savez("disk+halo+bkg+eg+cmb.npz", samples=samples_ )

