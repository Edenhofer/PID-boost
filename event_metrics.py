from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse

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


def stats():
    stat = {}
    cuts = np.arange(0, 1, 0.1)
    for p in particle_list:
        stat[p] = {'tpr': [], 'fpr': []}
        for cut in cuts:
            stat[p]['tpr'] += [data[p][(data[p]['isSignal'] == 1) & (data[p][particleID_list[p]] > cut)].size / data[p][data[p]['isSignal'] == 1].size]
            stat[p]['fpr'] += [data[p][(data[p]['isSignal'] == 0) & (data[p][particleID_list[p]] > cut)].size / data[p][(data[p]['isSignal'] == 0) | (data[p][particleID_list[p]] > cut)].size]
            print('Particle %10s has a tpr of %6.6f and a False Positive Rate (FPR) of %6.6f with a cut of %4.4f'%(p, stat[p]['tpr'][-1], stat[p]['fpr'][-1], cut))

    plt.plot(stat['K+']['tpr'], stat['K+']['fpr'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()


def confusion_graph(nbins=50):
    for d in detector_list:
        plt.suptitle('Binned pidLogLikelihood for detector %s'%(d))
        for i, p in enumerate(particle_list):
            for i_2, p_2 in enumerate(particle_list):
                plt.subplot(len(particle_list), len(particle_list), i*len(particle_list)+i_2+1)
                plt.title('Identified %s as %s'%(p, p_2))
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + d + '__bc'
                data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

        plt.show()


# Base definitions of stable particles and detector data
particle_list = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleID_list = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
detector_list = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']

# Read in all the particle's information into a dictionary of panda frames
data = {}
for p in particle_list:
    data[p] = rpd.read_root(p + '.root')

parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.')
parser.add_argument('--stats', dest='run_stats', action='store_true', default=False, help='Print out and visualize some statistics (default: False)')
parser.add_argument('--confusion-graph', dest='run_confusion_graph', action='store_true', default=False, help='Plot a matrix of binned likelihoods graphs (default: False)')

args = parser.parse_args()
if args.run_stats:
    stats()
if args.run_confusion_graph:
    confusion_graph()
