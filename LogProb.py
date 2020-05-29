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

def multifreq(param):
	
	R_disk, h_disk, j_disk, a_disk, R_halo, j_halo, a_halo, T_1420, T_820, T_600, T_408, T_150 = param
	

	T_bkg = param[7::]
	
	lnL = 0
	for i in range(len(T_bkg)):

		model = MD.Spheroid(l, b, R_disk, h_disk)*j_disk*(freqs[i]/(1e9))**(-a_disk) + MD.LineOfSightHalo(l, b, d, R_halo)*j_halo*(freqs[i]/(1e9))**(-a_halo)
		residuals = data[i] - (model)*(c**2)/(2*k*(freqs[i]**2)) - T_bkg[i]

		lnL += np.sum(np.log(1/(np.sqrt(2*np.pi)*errs[i]))) - np.sum((residuals**2)/(2*errs[i]**2))


	return lnL

def lnprob_multi(param, lower, upper):

	param = np.array(param)
	lower = np.array(lower)
	upper = np.array(upper)

	lp = lnprior(param, lower, upper)

	if not np.isfinite(lp):
		return -np.inf

	return lp + multifreq(param)

##############################################################################

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

## sph + halo #

#def diskhalo_TRIS(param, nu):
#	    
#	R_disk, h_disk, j_disk, R_halo, j_halo, T_bkg = param
#	sph1 = MD.Spheroid(TRIS_l[nu], TRIS_b[nu], R_disk, h_disk)*j_disk
#	halo1 = MD.LineOfSightHalo(TRIS_l[nu], TRIS_b[nu], d, R_halo)*j_halo
#	residuals = TRIS_Tb[nu] - (sph1 + halo1)*(c**2)/(2*k*(nu**2)) - T_CMB - T_bkg


#	lnL = np.sum(np.log(1/(np.sqrt(2*np.pi)*TRIS_Tberrs[nu]))) - np.sum((residuals**2)/(2*TRIS_Tberrs[nu]**2))
#	return lnL


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
'''
### ln(likelihood) for disk only model, including T_eg and T_CMB and T_bkg ###


### ln(likelihood) for halo model, including T_eg and T_CMB and T_bkg ###
def halobkg(param, nu, l, b, T_sky):
    
	    
	R_halo = param[0]
	j_halo = param[1]
	T_bkg = param[2]

	# residuals = T_res = T_sky - T_eg - T_CMB - T_disk - T_halo
	T_skysub = T_sky - T_eg - T_CMB - T_bkg
	residuals = T_skysub - (MD.LineOfSightHalo(l, b, d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))
	residuals[idx_exb] = None

	neg_res_idx = np.argwhere(residuals<=0)

	if len(neg_res_idx)<10:
		return -np.inf

	
	neg_res = residuals[neg_res_idx]

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL

### ln(likelihood) for disk + halo model, including T_eg and T_CMB###
def diskhalo(param, nu, l, b, T_sky):
    
	    
	R_disk = param[0]
	h_disk = param[1]
	j_disk = param[2]
	R_halo = param[3]
	j_halo = param[4]

	# residuals = T_res = T_sky - T_eg - T_CMB - T_disk - T_halo
	T_skysub = T_sky - T_eg - T_CMB
	residuals = T_skysub - (MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk + MD.LineOfSightHalo(l, b, d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))
	residuals[idx_exb] = None

	neg_res_idx = np.argwhere(residuals<=0)

	if len(neg_res_idx)<10:
		return -np.inf

	
	neg_res = residuals[neg_res_idx]

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL


### ln(likelihood) for disk + halo + uniform background model, including T_eg and T_CMB ###


### ln(likelihood) for disk + halo + uniform background model, excluding T_eg and T_CMB ###
def diskhalobkg_nocmb(param, nu, l, b, T_sky):

	R_disk = param[0]
	h_disk = param[1]
	j_disk = param[2]
	R_halo = param[3]
	j_halo = param[4]
	T_bkg = param[5]

	# residuals = T_res = T_sky - T_disk - T_halo - T_bkg
	T_skysub = T_sky - T_bkg
	residuals = T_skysub - (MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk + MD.LineOfSightHalo(l, b, d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))
	residuals[idx_exb] = None

	neg_res_idx = np.argwhere(residuals<=0)

	if len(neg_res_idx)<10:
		return -np.inf

	
	neg_res = residuals[neg_res_idx]

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	D = stats.kstest(neg_res2.T, 'norm')[0]
	n = len(neg_res2)
	
	idx = np.argmin(np.abs(nD-(np.sqrt(n)*D)))
	L = PDF[idx] 

	#L = stats.kstest(neg_res2.T, 'norm')[1]


	lnL = np.log(L)
	return lnL



##################################################################################################################

# Functions specific for analyzing TRIS data. Data formatted for these functions is stored in TRIS_vals.py

##################################################################################################################


### ln(likelihood) for disk + halo model ###
def diskhalo_TRIS(param, nu):
	    
	R_disk, h_disk, j_disk, R_halo, j_halo = param

	# residuals = T_res = T_gal - T_disk - T_halo
#	residuals = TRIS_Tgal[nu] - (MD.LineOfSightDisk(TRIS_l[nu], TRIS_b[nu], d, R_disk, h_disk)*j_disk + MD.LineOfSightHalo(TRIS_l[nu], TRIS_b[nu], d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))

	residuals = TRIS_Tgal[nu] - (MD.Spheroid(TRIS_l[nu], TRIS_b[nu], R_disk, h_disk)*j_disk + MD.LineOfSightHalo(TRIS_l[nu], TRIS_b[nu], d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))

	# reject models where any of the negative residuals are greater than 3*sigma in order to prevent over fitting
	if np.any(residuals < -3*TRIS_Tbsig[nu]):
		return -np.inf

	lnL = -np.trapz(np.abs(residuals), x=TRIS_ra[nu])
	return lnL

def spheroids_TRIS(param, nu):
	    
	R_disk, h_disk, j_disk, R_halo, h_halo, j_halo = param

	# residuals = T_res = T_gal - T_disk - T_halo
	disk = MD.Spheroid(TRIS_l[nu], TRIS_b[nu], R_disk, h_disk)*j_disk
	halo = MD.Spheroid(TRIS_l[nu], TRIS_b[nu], R_halo, h_halo)*j_halo

	residuals = TRIS_Tgal[nu] - (disk+halo)*(c**2)/(2*k*(nu**2)) 

	lnL = -np.trapz(np.abs(residuals[cygxmask]), x=TRIS_ra[nu][cygxmask])
	return lnL


### ln(likelihood) for halo model ###
def halo_TRIS(param, nu):
	    
	R_halo = param[0]
	j_halo = param[1]

	# residuals = T_res = T_gal - T_disk - T_halo

	residuals = TRIS_Tgal[nu] - (MD.LineOfSightHalo(TRIS_l[nu], TRIS_b[nu], d, R_halo)*j_halo)*(c**2)/(2*k*(nu**2))

	# reject models where any of the negative residuals are greater than 3*sigma in order to prevent over fitting
	if np.any(residuals < -3*TRIS_Tbsig[nu]):
		return -np.inf

	lnL = -np.trapz(np.abs(residuals), x=TRIS_ra[nu])
	return lnL


### ln(likelihood) for disk model ###



### lnL for a spheroid+halo model, including a spectral index to analyze both frequency bands simultaneously 

def multi_TRIS(param, nu1, nu2):

	R_disk, h_disk, j_disk, a_disk, R_halo, j_halo, a_halo = param

	m_disk = (nu2/nu1)**a_disk
	m_halo = (nu2/nu1)**a_halo

	sph1 = MD.Spheroid(TRIS_l[nu1], TRIS_b[nu1], R_disk, h_disk)*j_disk
	halo1 = MD.LineOfSightHalo(TRIS_l[nu1], TRIS_b[nu1], d, R_halo)*j_halo

	res1 = TRIS_Tgal[nu1] - (sph1 + halo1)*(c**2)/(2*k*(nu1**2))

	res2 = TRIS_Tgal[nu2] - (sph1*m_disk + halo1*m_halo)*(c**2)/(2*k*(nu1**2))

	lnL = -(np.trapz(np.abs(res1), x=TRIS_ra[nu1]) + np.trapz(np.abs(res2), x=TRIS_ra[nu1]))
	return lnL

'''


