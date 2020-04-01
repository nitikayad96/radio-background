import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import emcee
from scipy import stats
import corner
import sys
from scipy.optimize import curve_fit as cf

import ModelDefinitions as MD
import LogProb as LP
from const import *

# read map, and make sure maps are in Kelvin
map_1420 = (hp.read_map('../STOCKERT+VILLA-ELISA_1420MHz_1_256.fits'))/1000 - T_CMB - T_eg
aguess = [0.05, 10.*d, 0.3*d, 2.5e-41, 4*d, 4.e-42] # this matches the data well
nsid = NSIDE_sky

# uncomment for simulated map
#map_1420 = MD.sim_map(*aguess,noise=0.1)
#nsid = NSIDE_dg

# for displaying map
mm = map_1420.copy()
idx_exb = hp.query_strip(nsid, np.deg2rad(90-b_mask), np.deg2rad(90+b_mask))
mm[idx_exb] = None
hp.mollview(mm)
plt.show()

# prepare spherical harmonics from map and plot
data = MD.prep_data(map_1420)
Alm_data,sig = data
plt.plot(Alm_data,label='Data')

# prepare model and plot
model = LP.model(np.arange(len(Alm_data)), *aguess)
plt.plot(model,label='Model')
plt.plot(np.sqrt(sig),label='Sigma')
plt.legend()
plt.show()

# save lm_idx - this selects spherical harmonics to use in fit
#wrs = np.where(np.abs(model)>0.0001)
#print len(wrs[0])
#np.save('lm_idx.npy',wrs)

# run curve_fit
#popt, pcov = cf(LP.model,np.arange(len(Alm_data)),Alm_data,p0=aguess,sigma=sig,bounds=[[-3.,d,0.0001*d,0.,d,0.],[3.,100.*d,100.*d,1e-37,100.*d,1e-37]])
#
#for i in range(6):
#    if i==1:
#        print popt[i]/d,np.sqrt(pcov[i,i])/d
#    elif i==2:
#        print popt[i]/d,np.sqrt(pcov[i,i])/d
#    elif i==4:
#        print popt[i]/d,np.sqrt(pcov[i,i])/d
#    else:
#        print popt[i],np.sqrt(pcov[i,i])
#        
#sys.exit()

### set up priors for each MCMC run ###
# You change: value of priors #
def lnprior(param):

    T_bkg = param[0]
    R_disk = param[1]
    h_disk = param[2]
    j_disk = param[3]
    #R_halo = param[4]
    #j_halo = param[5]
    
    
    #if d < R_disk < 50*d and 0 < h_disk < 5.*d and 1e-44 < j_disk < 1e-37 and R_halo > 0 and 1e-44 < j_halo < 1e-37:
    if d < R_disk < 50.*d and 0. < h_disk < 5.*d and 1e-44 < j_disk < 1e-37:
        return 0.0
    
    return -np.inf


### full log probability ###
# You change: likelihood function, chosen from LogProb #

def lnprob(param, d1, d2):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    lpp = lp + LP.lik_disk(param, (d1, d2))
    if np.isnan(lpp):
        return -np.inf
    return lpp

### initialize walkers ###
# You change: ndim, nwalkers, param_init values
ndim = 4
nwalkers = 20
param_init = aguess[0:4]

init = [param_init]
for i in range(nwalkers-1):
    #start walkers such that all values are positive and mean multiplicative factor is 1
    vary = np.abs(np.random.randn(ndim)/10.+1)
    init.append([param_init[i]*vary[i] for i in range(ndim)])
    
init = np.array(init)

for ii in init:
    print 'INITIAL prob',ii,lnprob(ii,*data)

### set up sampler and run MCMC ###
# You change: nsteps, filename
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data), threads=1)

SAMPLER = sampler.run_mcmc(init, 2000)

samples = np.array(sampler.chain)
samples[:,:,1:3] /= d
#samples[:,:,4] /= d

print('Average acceptance fraction: ', np.mean(sampler.acceptance_fraction))

np.savez("output.npz", samples=samples)

