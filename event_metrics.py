import matplotlib.pyplot as plt
import numpy as np
import root_pandas as rpd
import scipy
import scipy.stats

import pdg


def basf2_Code(particle):
    """Return the pdgCode in a basf2 compatible way with escaped special characters.

    Args:
        particle: The name of the particle which should be translated.

    Returns:
        Return the escaped pdgCode.

    """
    r = pdg.from_name(particle)
    if r > 0:
        return str(r)
    elif r < 0:
        return '__mi' + str(abs(r))
    else:
        raise ValueError('Something unexpected happened while converting the input to an escaped pdgCode.')


# Base definitions of stable particles and detector data
particle_list = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleID_list = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
detector_list = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']

# Read in all the particle's information into a dictionary of panda frames
data = {}
for p in particle_list:
    data[p] = rpd.read_root(p + '.root')
    print(data[p].keys())

particleID_cut = 0.1
stat = {}
for p in particle_list:
    stat[p] = {}
    stat[p]['sensitivity'] = data[p][(data[p]['isSignal'] == 1) & (data[p][particleID_list[p]] > particleID_cut)].size / data[p][data[p]['isSignal'] == 1].size
    stat[p]['fpr'] = data[p][(data[p]['isSignal'] == 0) & (data[p][particleID_list[p]] > particleID_cut)].size / data[p][(data[p]['isSignal'] == 0) | (data[p][particleID_list[p]] > particleID_cut)].size
    print('Particle %10s has a sensitivity of %6.6f and a False Positive Rate (FPR) of %6.6f'%(p, stat[p]['sensitivity'], stat[p]['fpr']))

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

