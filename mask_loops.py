import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from tqdm import tqdm


# angle between two points on sphere
def ang(l1,b1,l2,b2):

    # convert b into usual theta
    t1 = np.pi/2.-b1
    t2 = np.pi/2.-b2

    # do angle
    dot_prod = np.sin(t1)*np.sin(t2)*np.cos(l1-l2)+np.cos(t1)*np.cos(t2)
    return np.abs(np.arccos(dot_prod))
    

# read in map
test_map = hp.read_map("haslam408_dsds_Remazeilles2014.fits")
nside = hp.get_nside(test_map)

# define continuum loops - Vidal+15
l = np.asarray([329.0,100.0,124.0,315.0,127.0,120.5])*np.pi/180.
b = np.asarray([17.5,-32.5,15.5,48.5,18.0,30.8])*np.pi/180.
r = np.asarray([58.0,45.5,32.5,19.8,67.2,72.4])*np.pi/180.

# for masking
width = 15.*np.pi/180. # rad
silly = False

# get (l,b) for all pixels in map
ipix = np.arange(len(test_map))

# do masking
print("Masking...")

if silly:
    print("Likely not to work...")
    masks = np.zeros(len(ipix))
    bmap,lmap = hp.pix2ang(nside,ipix) # in rad
    for i in tqdm(range(len(ipix))):
        for j in range(1):
            
            # use angle between pixel and center of loop
            if np.abs(ang(lmap[i],bmap[i],l[j],b[j])-r[j])<width/2.:
                masks[i] = 1

else:
    masks = np.zeros((len(l),len(ipix)))
    for j in range(len(r)):
        vec = hp.ang2vec(np.pi/2.-b[j],l[j])
        masks[j,hp.query_disc(nside,vec=vec,radius=r[j]+width/2.)] = 1
        masks[j,hp.query_disc(nside,vec=vec,radius=r[j]-width/2.)] = 0        

    for j in range(1,len(r)):
        masks[0,:] += masks[j,:]
        masks[0,:][masks[0,:]!=0] = 1
        
print ("Done!")

test_map[masks[0,:]==1] = hp.UNSEEN

hp.mollview(test_map,coord="G",title="Test map (Haslam DSDS)",min=10.,max=80.,unit='K')
plt.show()
