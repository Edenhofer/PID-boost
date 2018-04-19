import matplotlib.pyplot as plt
import numpy as np
import root_pandas as rpd
import scipy
import scipy.stats

import pdg


def basf2_Code(particle):
    """Return the pdgCode in a basf2 compatible way with escaped special characters.

    Args:
        particle: The particle name which should be translated.

    Returns:
        Return the escpaed pdgCode.

    """
    r = pdg.from_name(particle)
    if r > 0:
        return str(r)
    elif r < 0:
        return '__mi' + str(abs(r))
    else:
        raise ValueError('Something unexepcted happened while converting to pdgCode.')


# Base definitions of stable particles and detector data
particle_list = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleID_list = ['kaonID', 'pionID','electronID',  'muonID', 'protonID', 'deuteronID']
detector_list = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']

# Read in all the particle's information into a dictionary of panda frames
data = {}
for p in particle_list:
    data[p] = rpd.read_root(p + '.root')
    print(data[p].keys())

# Variables for altering the visual representation of the data
nbins = 50

for d in detector_list:
    plt.suptitle('Binned pidLogLikelihood for detector %s'%(d))
    for i, p in enumerate(particle_list):
        for i_2, p_2 in enumerate(particle_list):
            plt.subplot(len(particle_list), len(particle_list), i*len(particle_list)+i_2+1)
            plt.title('Identified %s as %s'%(p, p_2))
            column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + d + '__bc'
            print(data[p][column].describe())
            data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

    plt.show()
