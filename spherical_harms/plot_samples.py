import numpy as np, corner, matplotlib.pyplot as plt, sys, pylab

# make triangle plot
def triangle_plot(chain,labls,sfig='triangle.png'):

    samples = chain.reshape((-1,len(labls)))

    fig=corner.corner(samples,labels=labls,quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12})

    fig.savefig(sfig)

    
def plot_sampler(plotarr,labls,filename='sampler.png'):

    # reshape
    ndim = plotarr.shape[2]
    nsamps = plotarr.shape[1]
    nchains = plotarr.shape[0]
    #x = np.tile(np.tile(np.arange(nsamps),(nchains,1)),(ndim,1,1)).swapaxes(0,1).swapaxes(1,2)
    #pltdata = plotarr.reshape((nsamps*nchains,ndim)).swapaxes(0,1)
    #xdata = x.reshape((nsamps*nchains,ndim)).swapaxes(0,1)

    # plot each chain

    plt.figure()
    for param in range(ndim):
        pylab.subplot(ndim,1,param+1)
        for ch in range(nchains):
            plt.plot(plotarr[ch,:,param],'-')
        plt.title(labls[param])
        
    plt.savefig(filename)



d = np.load(sys.argv[1])['samples']
d = d[:,500:,:]
#labls = ['T_bkg','R_disk','h_disk','j_disk','R_halo','j_halo']
labls = ['T_bkg','R_disk','h_disk','j_disk']
triangle_plot(d,labls)
plot_sampler(d,labls)

