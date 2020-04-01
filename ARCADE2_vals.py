import healpy as hp
import numpy as np
from scipy import stats
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.io import fits

# load in data

ARC2_Tb = {}

with fits.open('arc2_3150_v19.fits') as arc3150:
    NSIDE = arc3150[0].header['NSIDE']
    ARC2_Tb[3150e6] = arc3150[1].data['TEMPERATURE']
    
with fits.open('arc2_3410_v19.fits') as arc3410:
    ARC2_Tb[3410e6] = arc3410[1].data['TEMPERATURE']

    
npix = hp.nside2npix(NSIDE)
pixels = np.arange(npix)

#find indicies of observations (regions wehre T != 0)

idx = {}
idx[3150e6] = np.where(ARC2_Tb[3150e6] != 0)
idx[3410e6] = np.where(ARC2_Tb[3410e6] != 0)

ARC2_Tobs = {}
ARC2_Tobs[3150e6] = ARC2_Tb[3150e6][idx[3150e6]]
ARC2_Tobs[3410e6] = ARC2_Tb[3410e6][idx[3410e6]]

# extragalactic brightness temps calculated in sky-brightness-model notebook

ARC2_Teg = {}
ARC2_Teg[3150e6] = 0.0103
ARC2_Teg[3410e6] = 0.0083

# get coordinates of obs

ARC2_l = {}
ARC2_b = {}

ARC2_l[3150e6], ARC2_b[3150e6] = hp.pix2ang(NSIDE, pixels[idx[3150e6]], nest=True, lonlat=True)
ARC2_l[3410e6], ARC2_b[3410e6] = hp.pix2ang(NSIDE, pixels[idx[3410e6]], nest=True, lonlat=True)

