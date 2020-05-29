import healpy as hp
import numpy as np
from scipy import stats
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u

#### TRIS Values #####
TRIS600txt = np.loadtxt('TRIS_absolute_600.txt', dtype=str)
TRIS820txt = np.loadtxt('TRIS_absolute_820.txt', dtype=str)

TRIS_Tb = {}
TRIS_Tb[0.6e9] = np.float_(TRIS600txt[:,1])
TRIS_Tb[0.82e9] = np.float_(TRIS820txt[:,1])

TRIS_Tbsig = {}
TRIS_Tbsig[0.6e9] = np.float_(TRIS600txt[:,2])
TRIS_Tbsig[0.82e9] = np.float_(TRIS820txt[:,2])

TRIS_Tbsys = {}
TRIS_Tbsys[0.6e9] = 0.066
TRIS_Tbsys[0.82e9] = np.sqrt(0.3**2 + 0.43**2)

TRIS_Tberrs = {}
TRIS_Tberrs[0.6e9] = np.sqrt(TRIS_Tbsys[0.6e9]**2 + TRIS_Tbsig[0.6e9]**2)
TRIS_Tberrs[0.82e9] = np.sqrt(TRIS_Tbsys[0.82e9]**2 + TRIS_Tbsig[0.82e9]**2)

TRIS_CMB = {}
TRIS_CMB[0.6e9] = 2.823
TRIS_CMB[0.82e9] = 2.783

TRIS_eg = {}
TRIS_eg[0.6e9] = 0.934
TRIS_eg[0.82e9] = 0.408

TRIS_ra = {}
TRIS_l = {}
TRIS_b = {}

RA600 = TRIS600txt[:,0]
sc600 = SkyCoord(ra=RA600, dec=42.0, unit=(u.hourangle,u.deg), frame='icrs')
sc600_gal = sc600.galactic

RA820 = TRIS820txt[:,0]
sc820 = SkyCoord(ra=RA820, dec=42.0, unit=(u.hourangle,u.deg), frame='icrs')
sc820_gal = sc820.galactic

TRIS_ra[0.6e9] = sc600.ra.value
TRIS_l[0.6e9] = sc600_gal.l.value
TRIS_b[0.6e9] = sc600_gal.b.value

TRIS_ra[0.82e9] = sc820.ra.value
TRIS_l[0.82e9] = sc820_gal.l.value
TRIS_b[0.82e9] = sc820_gal.b.value

TRIS_Tgal = {}

aref600 = np.ravel(np.where(RA600=='10h00m'))
Tgal_ref600 = 5.72
Tsky_ref600 = TRIS_Tb[0.6e9][aref600]
TRIS_Tgal[0.6e9] = TRIS_Tb[0.6e9] - Tsky_ref600 + Tgal_ref600

aref820 = np.ravel(np.where(RA820=='10h00m'))
Tgal_ref820 = 2.21
Tsky_ref820 = TRIS_Tb[0.82e9][aref820]
TRIS_Tgal[0.82e9] = TRIS_Tb[0.82e9] - Tsky_ref820 + Tgal_ref820

# Mask out Cygnus X region
idxmax = np.argmax(TRIS_Tb[0.6e9])
idxrange = np.arange(idxmax-8, idxmax+8)
TRIS600mask = np.array(TRIS_Tb[0.6e9])
TRIS600mask[idxrange]=0

cygxmask = np.where(TRIS600mask != 0)

