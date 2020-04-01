# Fitting in spherical harmonic space

The code here implements a fit to the radio sky in spherical harmonic space. The sky is modeled as the sum of the CMB (T_CMB), the known extragalactic component (T_EG), an unknown extragalactic component (T_BKG), a disk (T_disk), a spherical halo (T_halo), and noise. The noise consists of measurement error, and unmodeled structure that is assumed to follow a Gaussian random process.

T_CMB and T_EG are first subtracted from the data. The models that can then be run include:
1. T_BKG + T_disk
2. T_BKG + T_disk + T_halo

The code here includes:
* **const.py**: includes constants from other code, which are read into each module.
* **LogProb.py**: has a *model* function to calculate model spherical harmonics, and the likelihood functions *lik_disk* and *lik_disk_halo*.
* **ModelDefinitions.py**: the only addition here is the *prep_data* function to prepare the map spherical harmonics, and the *sim_map* function to make a simulated map.
* **runmcmc.py**: runs the mcmc, with a few other features. See comments in code.
* **plot_samples.py**: makes plots of the chains, and a triangle plot.
* **lm_idx.npy**: contains the idices of the spherical harmonic series that are used for fitting.



