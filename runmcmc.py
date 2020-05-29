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

LP.multifreq,

])

outfiles = np.array([

'multifreq.npz',

])


param_inits = np.array([

[2.3*d, d, 10**(-40.5), 0.8, 1.8*d, 10**(-41.5), 0.7, 0.1, 1, 3, 10, 30],

])

priors_lower = np.array([

[d, 0., 10**(-45), 0, 0, 10**(-45),0, 0, 0, 0, 0, 0],

])

priors_upper = np.array([

[10*d, 3*d, 10**(-40), 1, 10*d, 10**(-40), 1, 50, 50, 50, 50, 50],

])

#nus = 1e6*np.array([

#])

select = np.array([0])

print(modelprobs[select])

for i in select:


	# initialize walkers
	ndim = len(param_inits[i])
	nwalkers = 50

	init = [param_inits[i]]
	#print(LP.lnprior(init[0], priors_lower[i], priors_upper[i]))
	while len(init) < nwalkers:
		#print(len(init))
	    #start walkers such that all values are positive, std is x, and mean multiplicative factor is 1, and are within the priors

		check = -np.inf
		while not np.isfinite(check):
			vary = np.abs(np.random.randn(ndim)*0.25 + 1.)
			param0 = [param_inits[i][j]*vary[j] for j in range(ndim)]
			check = LP.lnprior(param0, priors_lower[i], priors_upper[i])
			
		
		init.append(param0)
	    
	init = np.array(init)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, LP.lnprob_multi, args=(priors_lower[i], priors_upper[i]), threads=8)

	SAMPLER = sampler.run_mcmc(init, 15000, progress=True)

	samples_ = np.array(sampler.chain)
	af = np.mean(sampler.acceptance_fraction)
	print(outfiles[i])
	print('Mean Acceptance Fraction: ' + str(af))
	np.savez(outfiles[i], samples=samples_, mean_acceptance_fraction = af)
	
	

