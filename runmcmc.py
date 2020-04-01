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

modelprobs = np.array([

LP.diskbkg,
LP.diskhalobkg,
LP.disk_TRIS,
LP.disk_TRIS,
LP.diskhalo_TRIS,
LP.diskhalo_TRIS,
LP.diskbkg_ARC2,
LP.diskbkg_ARC2,
LP.diskhalobkg_ARC2,
LP.diskhalobkg_ARC2,

])

outfiles = np.array([

'allsky_diskbkg.npz',
'allsky_diskhalobkg.npz',
'TRIS_disk_600.npz',
'TRIS_disk_820.npz',
'TRIS_diskhalo_600.npz',
'TRIS_diskhalo_820.npz',
'ARC2_diskbkg_3150.npz',
'ARC2_diskbkg_3410.npz',
'ARC2_diskhalobkg_3150.npz',
'ARC2_diskhalobkg_3410.npz',

])


param_inits = np.array([

[2.2*d, 0.5*d, 4.3*10**(-41), 0.5],
[2.2*d, 0.5*d, 4.3*10**(-41), 2.3*d, 6.5*10**(-42), 0.5],
[2.2*d, 0.5*d, 4.3*10**(-41), 0.5],
[2.2*d, 0.5*d, 4.3*10**(-41), 0.5],
[2.2*d, 0.5*d, 4.3*10**(-41), 2.3*d, 6.5*10**(-42), 0.5],
[2.2*d, 0.5*d, 4.3*10**(-41), 2.3*d, 6.5*10**(-42), 0.5],
[2.2*d, 0.5*d, 4.3*10**(-41), 0.1],
[2.2*d, 0.5*d, 4.3*10**(-41), 0.1],
[2.2*d, 0.5*d, 4.3*10**(-41), 2.3*d, 6.5*10**(-42), 0.1],
[2.2*d, 0.5*d, 4.3*10**(-41), 2.3*d, 6.5*10**(-42), 0.1],

])

priors_lower = np.array([

[d, 0., 10**(-45), 0.],
[d, 0., 10**(-45), d, 10**(-45), 0.],
[d, 0., 10**(-45),0],
[d, 0., 10**(-45),0],
[d, 0., 10**(-45), d, 10**(-45),0],
[d, 0., 10**(-45), d, 10**(-45),0],
[d, 0., 10**(-45), 0.],
[d, 0., 10**(-45), 0.],
[d, 0., 10**(-45), d, 10**(-45), 0.],
[d, 0., 10**(-45), d, 10**(-45), 0.],

])

priors_upper = np.array([

[10*d, 2*d, 10**(-40), 5.],
[10*d, 2*d, 10**(-40), 10*d, 10**(-40), 5.],
[10*d, 2*d, 10**(-40),2],
[10*d, 2*d, 10**(-40),2],
[10*d, 2*d, 10**(-40), 10*d, 10**(-40),2],
[10*d, 2*d, 10**(-40), 10*d, 10**(-40),2],
[10*d, 2*d, 10**(-40), 5.],
[10*d, 2*d, 10**(-40), 5.],
[10*d, 2*d, 10**(-40), 10*d, 10**(-40), 5.],
[10*d, 2*d, 10**(-40), 10*d, 10**(-40), 5.],

])

nus = 1e6*np.array([

1420.,
1420.,
600.,
820.,
600.,
820.,
3150.,
3410.,
3150.,
3410.

])

select = np.array([2,3,4,5])

print(modelprobs[select])

for i in select:


	# initialize walkers
	ndim = len(param_inits[i])
	nwalkers = 100

	init = [param_inits[i]]
	for j in range(nwalkers-1):
	    #start walkers such that all values are positive, std is x, and mean multiplicative factor is 1
	    vary = np.abs(np.random.randn(ndim) + 1.)
	    init.append([param_inits[i][j]*vary[j]*0.5 for j in range(ndim)])
	    
	init = np.array(init)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, LP.lnprob, args=(nus[i], priors_lower[i], priors_upper[i], modelprobs[i]), threads=8)

	SAMPLER = sampler.run_mcmc(init, 5000, progress=True)

	samples_ = np.array(sampler.chain)
	af = np.mean(sampler.acceptance_fraction)
	print(outfiles[i])
	print('Mean Acceptance Fraction: ' + str(af))
	np.savez(outfiles[i], samples=samples_, mean_acceptance_fraction = af)
	
	

