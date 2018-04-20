from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
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


def stats(cut_min=0., cut_max=1., ncuts=50):
    """Calculate, print and plot various values from statistics for further analysis and finally return some values.

    Args:
        cut_min: Lower bound of the cut (default: 0).
        cut_max: Upper bound of the cut (default: 1).
        ncuts: Number of cuts to perform on the interval (default: 50).

    Returns:
        A dictionary of dictionaries containing arrays themselfs.

        Each particle has an entry in the dictionary and each particle's dictionary has a dictionary of values from statistics for each cut:

            {
                'K+': {
                    'tpr': [True Positive Rate for each cut],
                    'fpr': [False Positive Rate for each cut],
                    'tnr': [True Negative Rate for each cut],
                    'ppv': [Positive Predicted Value for each cut]
                },
                ...
            }

    """
    stat = {}
    cuts = np.linspace(cut_min, cut_max, num=ncuts)
    for p in particles:
        stat[p] = {'tpr': [], 'fpr': [], 'tnr': [], 'ppv': []}
        for cut in cuts:
            stat[p]['tpr'] += [data[p][(data[p]['isSignal'] == 1) & (data[p][particleIDs[p]] > cut)].size / data[p][data[p]['isSignal'] == 1].size]
            stat[p]['fpr'] += [data[p][(data[p]['isSignal'] == 0) & (data[p][particleIDs[p]] > cut)].size / data[p][data[p]['isSignal'] == 0].size]
            stat[p]['tnr'] += [data[p][(data[p]['isSignal'] == 0) & (data[p][particleIDs[p]] < cut)].size / data[p][data[p]['isSignal'] == 0].size]
            stat[p]['ppv'] += [data[p][(data[p]['isSignal'] == 1) & (data[p][particleIDs[p]] > cut)].size / data[p][data[p][particleIDs[p]] > cut].size]

            if not np.isclose(stat[p]['fpr'][-1]+stat[p]['tnr'][-1], 1, atol=1e-2):
                print('VALUES INCONSISTENT: ', end='')

            print('Particle %10s: TPR=%6.6f; FPR=%6.6f; TNR=%6.6f; PPV=%6.6f; cut=%4.4f'%(p, stat[p]['tpr'][-1], stat[p]['fpr'][-1], stat[p]['tnr'][-1], stat[p]['ppv'][-1], cut))

        plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='True Positive Rate')
        plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
        plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='Positive Predicted Value')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        plt.title('Receiver Operating Characteristic (ROC) curve for %s identification'%(particle_formats[p]))
        plt.legend()
        plt.show()

    return stat


def confusion_graph(nbins=50):
    for d in detectors:
        plt.suptitle('Binned pidLogLikelihood for detector %s'%(d))
        for i, p in enumerate(particles):
            for i_2, p_2 in enumerate(particles):
                plt.subplot(len(particles), len(particles), i*len(particles)+i_2+1)
                plt.title('Identified %s as %s'%(particle_formats[p], particle_formats[p_2]))
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + d + '__bc'
                data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

        plt.show()


def epsilonPID_matrix(cut=0.3):
    epsilonPIDs = np.zeros(shape=(len(particles), len(particles)))
    for i, i_p in enumerate(particles):
        for j, j_p in enumerate(particles):
            # BUG: the deuterium code is not properly stored in the mcPDG variable and hence might lead to misleading visuals
            epsilonPIDs[i][j] = data[i_p][(data[i_p]['mcPDG'] == pdg.from_name(i_p)) & (data[i_p][particleIDs[j_p]] > cut)].size / data[i_p][data[i_p]['mcPDG'] == pdg.from_name(i_p)].size

    print("Confusion matrix:\n%s"%(epsilonPIDs))
    plt.imshow(epsilonPIDs, cmap='hot')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.colorbar()
    plt.title(r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut))
    plt.show()


# Base definitions of stable particles and detector data
particles = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleIDs = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
particle_formats = {'K+': r'$K^+$', 'pi+': r'$\pi^+$', 'e+': r'$e^+$', 'mu+': r'$\mu^+$', 'p+': r'$p^+$', 'deuteron': r'$d$'}
detectors = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']

# Read in all the particle's information into a dictionary of panda frames
data = {}
for p in particles:
    data[p] = rpd.read_root(p + '.root')

parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.')
parser.add_argument('--stats', dest='run_stats', action='store_true', default=False, help='Print out and visualize some statistics (default: False)')
parser.add_argument('--confusion-graph', dest='run_confusion_graph', action='store_true', default=False, help='Plot a matrix of binned likelihoods graphs (default: False)')
parser.add_argument('--epsilonPID-matrix', dest='run_epsilonPID_matrix', action='store_true', default=False, help='Plot the confusion matirx of every events (default: False)')

args = parser.parse_args()
if args.run_stats:
    stats()
if args.run_confusion_graph:
    confusion_graph()
if args.run_epsilonPID_matrix:
    epsilonPID_matrix()
