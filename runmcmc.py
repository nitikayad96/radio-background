import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import emcee
from scipy import stats
import corner

import ModelDefinitions as MD
import LogProb as LP
from const import *

import importlib
importlib.reload(LP)
importlib.reload(MD)


# make sure maps are in Kelvin
#map_1420 = (hp.read_map('STOCKERT+VILLA-ELISA_1420MHz_1_256.fits'))/1000
#map_1420_dg = hp.pixelfunc.ud_grade(map_1420, NSIDE_dg)
#idx_exb = hp.query_strip(NSIDE_dg, np.deg2rad(90-10), np.deg2rad(90+10))

# generate simulated maps for MCMC if desired
param_init = [2*d, 0.15*d, 10**(-40.62), 3.7*d, 10**(-41.56), 3.0]
R_disk, h_disk, j_disk, R_halo, j_halo, T_bkg = param_init
sim = MD.sim_map(T_bkg, R_disk, h_disk, j_disk, R_halo, j_halo, noise = 0.05)
sim[idx_exb] = None
sim_img = hp.mollview(sim)
plt.show()


### set up priors for each MCMC run ###
# You change: value of priors #
def lnprior(param):
    
    R_disk = param[0]
    h_disk = param[1]
    j_disk = param[2]
    R_halo = param[3]
    j_halo = param[4]
    T_bkg = param[5]
    
    if d < R_disk < 20*d and 0 < h_disk < 5*d and 1e-42 < j_disk < 1e-39 and d < R_halo < 50*d and 1e-43 < j_halo < 1e-38 and 0 <= T_bkg <= 20:
        return 0.0

    else:
    	return -np.inf


### full log probability ###
# You change: likelihood function, chosen from LogProb #

def lnprob(param, nu, l, b, T_sky, T_eg, idx_exb):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + LP.diskhalobkg_nocmb(param, nu, l, b, T_sky, T_eg, idx_exb)

### initialize walkers ###
# You change: ndim, nwalkers, param_init values
ndim = 6
nwalkers = 20
#param_init = [2*d, 0.15*d, 10**(-40.62), 3.7*d, 10**(-41.56), 3.1]

init = [param_init]
for i in range(nwalkers-1):
    #start walkers such that all values are positive and mean multiplicative factor is 1
    vary = np.abs(np.random.randn(ndim)+1)
    init.append([param_init[i]*vary[i] for i in range(ndim)])
    
init = np.array(init)

### set up sampler and run MCMC ###
# You change: nsteps, sky map, filename
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(nu, l, b, sim, T_eg, idx_exb), threads=8)

SAMPLER = sampler.run_mcmc(init, 500, progress=True)

samples_ = np.array(sampler.chain)
print('Average acceptance fraction: ', np.mean(sampler.acceptance_fraction))

np.savez("disk+halo+bkg_ksDsim.npz", samples=samples_ )

