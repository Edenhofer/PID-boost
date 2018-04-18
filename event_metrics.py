import root_pandas as rpd
import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

nbins = 50

particles_of_interest = ['K-', 'pi+']   # Pay extra attention to use the same order as for the preprocessing step otherwise the epsilon_PID matrix will be off
detector_list = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']
particleID_list = ['kaonID', 'pionID']

data = rpd.read_root('sample.root')
#print(data.describe())

for d in detector_list:
    plt.suptitle('Binned pidLogLikelihood for detector %s'%(d))
    for i, p in enumerate(particles_of_interest):
        for i_2, p_2 in enumerate(particles_of_interest):
            plt.subplot(len(particles_of_interest), len(particles_of_interest), i*len(particles_of_interest)+i_2+1)
            plt.title('%s as %s'%(p, p_2))
            column = 'pidLogLikelihoodValueExpert_' + d + 'Detector_daughter' + str(i) + 'asDaughter' + str(i_2)
            print(data[column].describe())
            data[column].hist(bins=nbins)

    plt.show()

