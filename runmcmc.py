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
LP.diskhalo_TRIS

])

outfiles = np.array([

'multifreq_sim_lstsq.npz',
'TRIS600_test3.npz'

])


param_inits = np.array([

[ 2.02751922e+00,  7.00000000e-01, -4.04481897e+01,  6.46896340e-01, 3e+00, -4.14516223e+01,  9.80776467e-01, 1.65798469e-01, 1.16636636e+00,  3.37031922e+00,  3.94665343e+00,  7e+00],
[2.2, 0.9, -40.1, 2, -41.5, 2]

])

priors_lower = np.array([

[1, 0., -42, 0, 0, -42, 0, 0, 0, 0, 0, 0],
[1, 0, -42, 0, -42, 0]

])

priors_upper = np.array([

[10, 3, -40, 1, 10, -40, 1, 10, 10, 10, 10, 10],
[10, 2, -40, 10, -40, 5]
])


# select index of which model to run
select = np.array([0])

print(modelprobs[select][0])

for i in select:
	print(i)

	# initialize walkers
	ndim = len(param_inits[i])
	nwalkers = 25

	init = [param_inits[i]]
	#print(LP.lnprior(init[0], priors_lower[i], priors_upper[i]))
	print('Initializing Walkers')
	while len(init) < nwalkers:
		#print(len(init))
	    #start walkers such that all values are positive, std is x, and mean multiplicative factor is 1, and are within the priors

		check = -np.inf
		while not np.isfinite(check):
			vary = np.abs(np.random.randn(ndim)*0.1 + 1.)
			param0 = [param_inits[i][j]*vary[j] for j in range(ndim)]
			check = LP.lnprior(param0, priors_lower[i], priors_upper[i])
			
		
		init.append(param0)
	    
	init = np.array(init)
	print('Initialization Complete')

	if modelprobs[select][0] == LP.multifreq:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, LP.lnprob_multi, args=(priors_lower[i], priors_upper[i],True), threads=8)

	else: 
		sampler = emcee.EnsembleSampler(nwalkers, ndim, LP.lnprob, args=(0.6e9, priors_lower[i], priors_upper[i], modelprobs[select][0]), threads=8)

	SAMPLER = sampler.run_mcmc(init, 15000, progress=True)

	samples_ = np.array(sampler.chain)
	af = np.mean(sampler.acceptance_fraction)
	

	np.savez(outfiles[i], samples=samples_, mean_acceptance_fraction = af)

	print(outfiles[i])
	print('Mean Acceptance Fraction: ' + str(af))

	autocorr_time = np.mean(sampler.get_autocorr_time())
	print('Mean Autocorrelation Time: {0:.3f} steps'.format(autocorr_time))

	
	

