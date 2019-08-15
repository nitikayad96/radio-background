'''
Define log of likelihood function. 
Make sure all sky maps are in Kelvin 
'''

import ModelDefinitions as MD
import numpy as np
from scipy import stats
from const import *
from TRIS_vals import *

### ln(likelihood) for disk only model, including T_eg and T_CMB and T_bkg ###
def diskbkg(param, nu, l, b, T_sky): 

	R_disk = param[0]
	h_disk = param[1]
	j_disk = param[2]
	T_bkg = param[3]

	# residuals = T_res = T_sky - T_eg - T_CMB - T_bkg - T_disk
	T_skysub = T_sky - T_eg - T_CMB - T_bkg
	residuals = T_skysub - MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2))
	residuals[idx_exb] = None

	neg_res_idx = np.argwhere(residuals<=0)

	if len(neg_res_idx)<10:
		return -np.inf

	
	neg_res = residuals[neg_res_idx]
	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL

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

### ln(likelihood) for disk + halo model, including T_eg and T_CMB###
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


### ln(likelihood) for disk + halo model, including T_eg and T_CMB###
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

### ln(likelihood) for disk + halo + uniform background model, including T_eg and T_CMB ###
def diskhalobkg(param, nu, l, b, T_sky):
    
	R_disk = param[0]
	h_disk = param[1]
	j_disk = param[2]
	R_halo = param[3]
	j_halo = param[4]
	T_bkg = param[5]

	# residuals = T_res = T_sky - T_eg - T_CMB - T_disk - T_halo - T_bkg
	T_skysub = T_sky - T_eg - T_CMB - T_bkg
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

