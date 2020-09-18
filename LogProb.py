'''
Define log of likelihood function. 
Make sure all sky maps are in Kelvin 
'''

import ModelDefinitions as MD
import numpy as np
from scipy import stats
from const import *
from multifreq_data import *

#from TRIS_vals import *
#from ARCADE2_vals import *


############### Generalized prior and logprob functions #####################

# generalized prior function
def lnprior(param, lower, upper):

	param = np.array(param)
	lower = np.array(lower)
	upper = np.array(upper)

	l = lower < param
	u = param < upper

	if (np.sum(l) + np.sum(u)) != 2*len(param):
		return -np.inf
	
	else:
		return 0.0

# generalized logprob function
def lnprob(param, nu, lower, upper, model):

	param = np.array(param)
	lower = np.array(lower)
	upper = np.array(upper)

	lp = lnprior(param, lower, upper)

	if not np.isfinite(lp):
		return -np.inf

	return lp + model(param,nu)

############################ Multi-Freq Fit #################################

# residual function
def resid(param):

	R_disk, h_disk, j_disk, a_disk, R_halo, j_halo, a_halo, T_1420, T_820, T_600, T_408, T_150 = param
	R_disk *= d
	h_disk *= d
	R_halo *= d
	j_disk = 10.**j_disk
	j_halo = 10.**j_halo
	T_bkg = param[7::]
	resids = []
	for i in range(len(T_bkg)):

		model = MD.Spheroid(l, b, R_disk, h_disk)*j_disk*(freqs[i]/(1e9))**(-a_disk) + MD.LineOfSightHalo(l, b, R_halo)*j_halo*(freqs[i]/(1e9))**(-a_halo)

		resids.append(np.log10(data[i]/((model)*(c**2)/(2*k*(freqs[i]**2)) + T_bkg[i])))

	retvals = np.zeros(len(T_bkg)*len(resids[0]))
	for i in range(len(T_bkg)):
		retvals[i*len(resids[0]):(i+1)*len(resids[0])] = resids[i]

	return retvals

        

# function to return data and model
def return_stuff(param):

	R_disk, h_disk, j_disk, a_disk, R_halo, j_halo, a_halo, T_1420, T_820, T_600, T_408, T_150 = param
	j_disk = 10.**j_disk
	j_halo = 10.**j_halo
	R_disk *= d
	h_disk *= d
	R_halo *= d

	T_bkg = param[7::]
	returns = []
	for i in range(len(T_bkg)):

		model = MD.Spheroid(l, b, R_disk, h_disk)*j_disk*(freqs[i]/(1e9))**(-a_disk) + MD.LineOfSightHalo(l, b, R_halo)*j_halo*(freqs[i]/(1e9))**(-a_halo)
		returns.append((data[i],(model)*(c**2)/(2*k*(freqs[i]**2)) + T_bkg[i]))

	return returns


########## Simulated Data ##################

p_sim = [2.5e+00,  5e-01, -4.05e+01,  6.55e-01, 2.3e+00, -4.15e+01,  9.2e-01,  2.2e-01, 1.5e+00,  5e+00,  7e+00,  9e+00]
d_sim = return_stuff(p_sim)
data_sim = np.array([d_sim[0][1], d_sim[1][1], d_sim[2][1], d_sim[3][1], d_sim[4][1]])
noise = np.random.randn(data_sim.shape[0], data_sim.shape[1])*0.01 + 1
data_noisy = np.multiply(data_sim, noise)

###############################################
        

def multifreq(param, sim):
	
	R_disk, h_disk, j_disk, a_disk, R_halo, j_halo, a_halo, T_1420, T_820, T_600, T_408, T_150 = param
	j_disk = 10.**j_disk
	j_halo = 10.**j_halo
	R_disk *= d
	h_disk *= d
	R_halo *= d
	
	T_bkg = param[7::]

	lnL = 0
	for i in range(len(T_bkg)):

		model = (MD.Spheroid(l, b, R_disk, h_disk)*j_disk*(freqs[i]/(1e9))**(-a_disk) + MD.LineOfSightHalo(l, b, R_halo)*j_halo*(freqs[i]/(1e9))**(-a_halo))*(c**2)/(2*k*(freqs[i]**2)) + T_bkg[i]

		if sim==False:
			data_ = data[i]
			

		if sim==True:
			data_ = data_sim[i]
			

		lnL += -np.sum(np.abs(np.log10(data_/model)))
			

#		residuals = data_ - model
#		lnL += np.sum(np.log(1/(np.sqrt(2*np.pi)*errs[i]))) - np.sum((residuals**2)/(2*errs[i]**2))


	return lnL

def lnprob_multi(param, lower, upper, sim):

	param = np.array(param)
	lower = np.array(lower)
	upper = np.array(upper)

	lp = lnprior(param, lower, upper)

	if not np.isfinite(lp):
		return -np.inf

	return lp + multifreq(param,sim)

############################## TRIS ###########################################


def TRISmodel(param,nu):

	R_disk, h_disk, j_disk, R_halo, j_halo, sig_sys = param
	j_disk = 10.**j_disk
	j_halo = 10.**j_halo
	R_disk *= d
	h_disk *= d
	R_halo *= d

	sph1 = MD.Spheroid(TRIS_l[nu], TRIS_b[nu], R_disk, h_disk)*j_disk
	halo1 = MD.LineOfSightHalo(TRIS_l[nu], TRIS_b[nu], R_halo)*j_halo

	model = (sph1 + halo1)*(c**2)/(2*k*(nu**2))  + T_CMB
	return model

TRIS_psim = [2e+00,  6e-01, -4.04e+01, 3e+00, -4.1e+01, 0]
TRIS_sim_ = TRISmodel(TRIS_psim, 600e6)
TRIS_noise = np.random.randn(len(TRIS_l[600e6]))*0.01 + 1
TRIS_sim = np.multiply(TRIS_sim_, TRIS_noise)
  
	
def diskhalo_TRIS(param, nu):
	    
	R_disk, h_disk, j_disk, R_halo, j_halo, sig_sys = param

	#residuals = TRIS_sim_ - TRISmodel(param,nu)
	residuals = TRIS_Tb[0.6e9] - TRISmodel(param,nu)

	err_tot = np.sqrt(TRIS_Tberrs[nu]**2 + sig_sys**2)
	lnL = np.sum(np.log(1/(np.sqrt(2*np.pi)*err_tot))) - np.sum((residuals**2)/(2*err_tot**2))
	return lnL

####################### All sky map models ###################################

## sph + bkg #

#def diskbkg(param, nu): 

#	R_disk, h_disk, j_disk, T_bkg = param

#	# residuals = T_res = T_sky - T_eg - T_CMB - T_bkg - T_disk
#	T_skysub = map_1420_dg - T_eg - T_CMB - T_bkg
#	residuals = T_skysub - MD.Spheroid(l, b, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2))
#	residuals[idx_exb] = None

#	neg_res_idx = np.argwhere(residuals<=0)

#	if len(neg_res_idx)<10:
#		return -np.inf

#	
#	neg_res = residuals[neg_res_idx]
#	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
#	L = stats.kstest(neg_res2.T, 'norm')[1]

#	lnL = np.log(L)
#	return lnL

## sph + halo + bkg #

#def diskhalobkg(param, nu, T_sky):

#	R_disk, h_disk, j_disk, R_halo, j_halo, T_bkg = param

#	# residuals = T_res = T_sky - T_eg - T_CMB - T_disk - T_halo - T_bkg
#	T_skysub = T_sky - T_eg - T_CMB - T_bkg
#	residuals = T_skysub - (MD.Spheroid(l, b, R_disk, h_disk)*j_disk + MD.LineOfSightHalo(l, b, d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))
#	residuals[idx_exb] = None

#	neg_res_idx = np.argwhere(residuals<=0)

#	if len(neg_res_idx)<10:
#		return -np.inf

#	neg_res = residuals[neg_res_idx]
#	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
#	L = stats.kstest(neg_res2.T, 'norm')[1]

#	lnL = np.log(L)
#	return lnL


#################################### TRIS ###########################################

## sph #

#def disk_TRIS(param, nu):

#	R_disk, h_disk, j_disk, err_sys = param

#	# residuals = T_res = T_gal - T_disk
#	sph1 = MD.Spheroid(TRIS_l[nu], TRIS_b[nu], R_disk, h_disk)*j_disk
#	residuals = TRIS_Tgal[nu] - (sph1)*(c**2)/(2*k*(nu**2))

#	err_tot = np.sqrt(TRIS_Tbsig[nu]**2 + err_sys**2)

#	lnL = np.sum(np.log(1/np.sqrt(2*np.pi*err_tot)) - np.sum((residuals**2)/(2*err_tot**2))
#	return lnL

# sph + halo #




################################ ARCADE2 #########################################

## sph + bkg #

#def diskbkg_ARC2(param, nu):

#	R_disk, h_disk, j_disk, T_bkg = param

#	# residuals = T_res = T_sky - T_eg - T_CMB - T_bkg - T_disk
#	T_skysub = ARC2_Tobs[nu] - ARC2_Teg[nu] - T_CMB - T_bkg
#	residuals = T_skysub - MD.Spheroid(ARC2_l[nu], ARC2_b[nu], R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2))

#	lnL = -np.abs(np.sum(residuals))
#	return lnL

## sph + halo + bkg #

#def diskhalobkg_ARC2(param, nu):

#	R_disk, h_disk, j_disk, R_halo, j_halo, T_bkg = param

#	# residuals = T_res = T_sky - T_eg - T_CMB - T_bkg - T_disk
#	T_skysub = ARC2_Tobs[nu] - ARC2_Teg[nu] - T_CMB - T_bkg
#	residuals = T_skysub - (MD.Spheroid(ARC2_l[nu], ARC2_b[nu], R_disk, h_disk)*j_disk + MD.LineOfSightHalo(ARC2_l[nu], ARC2_b[nu], d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))

#	lnL = -np.abs(np.sum(residuals))
#	return lnL


##############################################################################


