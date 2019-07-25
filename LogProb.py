'''
Define log of likelihood function. 
Make sure all sky maps are in Kelvin 
'''

import ModelDefinitions as MD
import numpy as np
from scipy import stats

### ln(likelihood) for disk only model, including T_eg and T_CMB ###
def disk(param, nu, l, b, T_sky, T_eg, idx_exb):

   	#idx_exb = latitudes of -10 < b < 10 that we wish to exclude
	T_CMB = 2.725
	c = 3e10
	k = 1.381e-16
	pc = 3.086e18
	d = 8e3*pc # distance from galactic center to sun ~8kpc    

	R_disk = param[0]
	h_disk = param[1]
	j_disk = param[2]

	# residuals = T_res = T_sky - T_eg - T_CMB - T_disk
	T_skysub = T_sky - T_eg - T_CMB
	residuals = T_skysub - MD.LineOfSightDisk(l, b, d, R_disk, h_disk)*j_disk*(c**2)/(2*k*(nu**2))
	residuals[idx_exb] = None

	neg_res_idx = np.argwhere(residuals<=0)
	neg_res = residuals[neg_res_idx]

	if len(neg_res)==0:
		return -np.inf

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL

### ln(likelihood) for disk + halo model, including T_eg and T_CMB ###
def diskhalo(param, nu, l, b, T_sky, T_eg, idx_exb):
    
   	#idx_exb = latitudes of -10 < b < 10 that we wish to exclude
	T_CMB = 2.725
	c = 3e10
	k = 1.381e-16
	pc = 3.086e18
	d = 8e3*pc # distance from galactic center to sun ~8kpc  
	    
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
	neg_res = residuals[neg_res_idx]

	if len(neg_res)==0:
		return -np.inf

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL

### ln(likelihood) for disk + halo + uniform background model, including T_eg and T_CMB ###
def diskhalobkg(param, nu, l, b, T_sky, T_eg, idx_exb):
    
   	#idx_exb = latitudes of -10 < b < 10 that we wish to exclude
	T_CMB = 2.725  
	c = 3e10
	k = 1.381e-16
	pc = 3.086e18
	d = 8e3*pc # distance from galactic center to sun ~8kpc
    
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
	neg_res = residuals[neg_res_idx]

	if len(neg_res)==0:
		return -np.inf

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL

### ln(likelihood) for disk + halo + uniform background model, excluding T_eg and T_CMB ###
def diskhalobkg_nocmb(param, nu, l, b, T_sky, T_eg, idx_exb):
    
   	#idx_exb = latitudes of -10 < b < 10 that we wish to exclude
	c = 3e10
	k = 1.381e-16
	pc = 3.086e18
	d = 8e3*pc # distance from galactic center to sun ~8kpc

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
	neg_res = residuals[neg_res_idx]

	if len(neg_res)==0:
		return -np.inf

	neg_res2 = np.concatenate((neg_res, np.negative(neg_res)))
	L = stats.kstest(neg_res2.T, 'norm')[1]

	lnL = np.log(L)
	return lnL
