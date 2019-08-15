import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import emcee
from scipy import stats
import corner

import ModelDefinitions as MD
import LogProb as LP
from const import *
from TRIS_vals import *

import importlib
importlib.reload(LP)
importlib.reload(MD)

# set initial parameters
# You change: their value/length
param_init = [1.6*d, 0.5*d, 0.75*10**(-40), 3*d, 0.5*10**(-41)]

# make sure maps are in Kelvin
#map_1420 = (hp.read_map('STOCKERT+VILLA-ELISA_1420MHz_1_256.fits'))/1000
#map_1420_dg = hp.pixelfunc.ud_grade(map_1420, NSIDE_dg)

# If using TRIS data:
nu = 0.6e9

# generate simulated maps for MCMC if desired
#R_disk, h_disk, j_disk, R_halo, j_halo, T_bkg = param_init
#sim = MD.sim_map(T_bkg, R_disk, h_disk, j_disk, R_halo, j_halo, , noise = 0.05)
#sim[idx_exb] = None
#sim_img = hp.mollview(sim)
#plt.show()


### set up priors for each MCMC run ###
# You change: value of priors #
def lnprior(param):
    

    R_disk, h_disk, j_disk, R_halo, j_halo = param
    
    if d < R_disk < 20*d and 0 < h_disk < d and 1e-43 < j_disk < 1e-39 and d < R_halo < 20*d and 1e-43 < j_halo < 1e-39:
        return 0.0

    else:
    	return -np.inf


### full log probability ###
# You change: likelihood function, chosen from LogProb #

def lnprob(param, nu):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + LP.diskhalo_TRIS(param, nu)

### initialize walkers ###
# You change: nwalkers, can also change multiplicative parameter on std dev
ndim = len(param_init)
nwalkers = 100

init = [param_init]
for i in range(nwalkers-1):
    #start walkers such that all values are positive, std is x, and mean multiplicative factor is 1
    vary = np.abs(np.random.randn(ndim)*0.25 + 1.)
    init.append([param_init[i]*vary[i] for i in range(ndim)])
    
init = np.array(init)


### set up sampler and run MCMC ###
# You change: nsteps, sky map, filename
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(nu,), threads=8)

SAMPLER = sampler.run_mcmc(init, 4000, progress=True)

samples_ = np.array(sampler.chain)
print('Average acceptance fraction: ', np.mean(sampler.acceptance_fraction))

np.savez("TRIS_3sig_sph+halo.npz", samples=samples_ )

