from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
#import pandas as pd
import root_pandas as rpd
import scipy
import scipy.stats

import pdg


# Base definitions of stable particles and detector data
particles = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleIDs = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
particle_formats = {'K+': r'$K^+$', 'pi+': r'$\pi^+$', 'e+': r'$e^+$', 'mu+': r'$\mu^+$', 'p+': r'$p^+$', 'deuteron': r'$d$'}
detectors = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']
pseudo_detectors = ['all', 'default']
# Use the detector weights to exclude certain detectors, e.g. for debugging purposes
# Bare in mind that if all likelihoods are calculated correctly this should never improve the result
detector_weights = {d: 1. for d in detectors + pseudo_detectors}

# Read in all the particle's information into a dictionary of panda frames
data = {}
for p in particles:
    data[p] = rpd.read_root(p + '.root')


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


def stats(cut_min=0., cut_max=1., ncuts=50, cutting_columns=particleIDs):
    """Calculate, print and plot various values from statistics for further analysis and finally return some values.

    Args:
        cut_min: Lower bound of the cut (default: 0).
        cut_max: Upper bound of the cut (default: 1).
        ncuts: Number of cuts to perform on the interval (default: 50).
        cutting_columns: Dictionary which yields a column name for each particle on which basis the various statistics are calculated.

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
            stat[p]['tpr'] += [data[p][(data[p]['isSignal'] == 1) & (data[p][cutting_columns[p]] > cut)].size / data[p][data[p]['isSignal'] == 1].size]
            stat[p]['fpr'] += [data[p][(data[p]['isSignal'] == 0) & (data[p][cutting_columns[p]] > cut)].size / data[p][data[p]['isSignal'] == 0].size]
            stat[p]['tnr'] += [data[p][(data[p]['isSignal'] == 0) & (data[p][cutting_columns[p]] <= cut)].size / data[p][data[p]['isSignal'] == 0].size]
            stat[p]['ppv'] += [data[p][(data[p]['isSignal'] == 1) & (data[p][cutting_columns[p]] > cut)].size / data[p][data[p][cutting_columns[p]] > cut].size]

            if not np.isclose(stat[p]['fpr'][-1]+stat[p]['tnr'][-1], 1, atol=1e-2):
                print('VALUES INCONSISTENT: ', end='')

            print('Particle %10s: TPR=%6.6f; FPR=%6.6f; TNR=%6.6f; PPV=%6.6f; cut=%4.4f'%(p, stat[p]['tpr'][-1], stat[p]['fpr'][-1], stat[p]['tnr'][-1], stat[p]['ppv'][-1], cut))

        plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='True Positive Rate (ROC curve)')
        # Due to the fact that FPR + TNR = 1 the plot will simply show a straight line; Use for debugging only
        # plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
        plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='Positive Predicted Value')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        plt.title('%s identification'%(particle_formats[p]))
        plt.legend()
        plt.show()

    return stat


def epsilonPID_matrix(cut=0.2):
    """Calculate the epsilon_PID matrix for misclassifying particles, print the result and plot a heatmap.

    Args:
        cut: Position of the cut for the particleIDs.

    """
    epsilonPIDs = np.zeros(shape=(len(particles), len(particles)))
    for i, i_p in enumerate(particles):
        for j, j_p in enumerate(particles):
            # BUG: the deuterium code is not properly stored in the mcPDG variable and hence might lead to misleading visuals
            epsilonPIDs[i][j] = data[i_p][(data[i_p]['mcPDG'] == pdg.from_name(i_p)) & (data[i_p][particleIDs[j_p]] > cut)].size / data[i_p][data[i_p]['mcPDG'] == pdg.from_name(i_p)].size

    print("Confusion matrix:\n%s"%(epsilonPIDs))
    plt.imshow(epsilonPIDs)
    for (j, i), label in np.ndenumerate(epsilonPIDs):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.colorbar()
    plt.title(r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut))
    plt.show()


def mimic_ID(detector_weights=detector_weights, check=True):
    """Mimic the calculation of the particleIDs and compare them to their value provided by the analysis software.

    Args:
        detector_weights: Dictionary of detectors with the weights (default 1.) as values.
        check: Whether to assert the particleIDs if the detector weights are all 1.

    """
    if not all(v == 1. for v in detector_weights.values()):
        check = False

    for l in particles:
        # Calculate the accumulated logLikelihood and assess the relation to kaonID
        for p in particles:
            data[l]['accumulatedLogLikelihood' + basf2_Code(p)] = np.zeros(data[l][particleIDs[p]].size)
            # The following loop is equivalent to querying the 'all' pseudo-detector when using flat detector weights
            for d in detectors:
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
                # Fill up NaN values for detectors which do not yield a result
                # Since at least one detector must return a logLikelihood it is not possible that only NaN values lead to a probability of 1
                data[l]['accumulatedLogLikelihood' + basf2_Code(p)] += data[l][column].fillna(0) * detector_weights[d]

        # Calculate the particleIDs manually and compare them to the result of the analysis software
        data[l]['assumed_pionID'] = 1. / (1. + (data[l]['accumulatedLogLikelihood' + basf2_Code('K+')] - data[l]['accumulatedLogLikelihood' + basf2_Code('pi+')]).apply(np.exp))
        for p in (set(particles) - set(['pi+'])):
            # Algebraic trick to make exp(a)/(exp(a) + exp(b)) stable even for very small values of a and b
            data[l]['assumed_' + particleIDs[p]] = 1. / (1. + (data[l]['accumulatedLogLikelihood' + basf2_Code('pi+')] - data[l]['accumulatedLogLikelihood' + basf2_Code(p)]).apply(np.exp))

            if check:
                # Assert for equality of the manual calculation and analysis software's output
                npt.assert_allclose(data[l]['assumed_' + particleIDs[p]].values, data[l][particleIDs[p]].astype(np.float64).values, atol=1e-3)

    print('Successfully calculated the particleIDs using the logLikelihoods only.')


def bayes(priors=defaultdict(lambda: 1., {})):
    """Compute probabilities for particle hypothesis using a bayesian approach.

    Args:
        priors: Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.

    """
    for l in particles:
        # TODO: Use mimic_ID here to allow for weighted detector

        for p in particles:
            # Calculate the particleIDs manually and compare them to the result of the analysis software
            denominator = 0.
            for p_2 in particles:
                denominator += (data[l]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__spall__bc'] - data[l]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__spall__bc']).apply(np.exp) * priors[p_2]

            # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
            data[l]['bayes_' + particleIDs[p]] = priors[p] / denominator

    c = {k: 'bayes_' + v for k, v in particleIDs.items()}
    stats(cutting_columns=c)


def logLikelihood_by_particle(nbins=50):
    for d in detectors + pseudo_detectors:
        plt.suptitle('Binned pidLogLikelihood for detector %s'%(d))
        for i, p in enumerate(particles):
            for i_2, p_2 in enumerate(particles):
                plt.subplot(len(particles), len(particles), i*len(particles)+i_2+1)
                plt.title('Identified %s as %s'%(particle_formats[p], particle_formats[p_2]))
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + d + '__bc'
                data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

        plt.show()


def logLikelihood_by_detector(nbins=50):
    for p in particles:
        plt.suptitle('Binned pidLogLikelihood for particle %s'%(particle_formats[p]))
        for i, d in enumerate(detectors + pseudo_detectors):
            plt.subplot(2, len(detectors + pseudo_detectors), i+1)
            plt.title('Detector %s with signal'%(d))
            column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
            data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

            plt.subplot(2, len(detectors + pseudo_detectors), i+1+len(detectors + pseudo_detectors))
            plt.title('Detector %s with no signal'%(d))
            column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
            data[p][data[p]['isSignal'] == 0][column].hist(bins=nbins)

        plt.show()


parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.')
parser.add_argument('--stats', dest='run_stats', action='store_true', default=False, help='Print out and visualize some statistics (default: False)')
parser.add_argument('--logLikelihood-by-particle', dest='run_logLikelihood_by_particle', action='store_true', default=False, help='Plot the binned logLikelihood for each particle (default: False)')
parser.add_argument('--epsilonPID-matrix', dest='run_epsilonPID_matrix', action='store_true', default=False, help='Plot the confusion matirx of every events (default: False)')
parser.add_argument('--logLikelihood-by-detector', dest='run_logLikelihood_by_detector', action='store_true', default=False, help='Plot the binned logLikelihood for each detector (default: False)')
parser.add_argument('--mimic-ID', dest='run_mimic_ID', action='store_true', default=False, help='Mimic the calculation of the particle IDs using likelihoods (default: False)')
parser.add_argument('--bayes', dest='run_bayes', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis using bayes')
parser.add_argument('--bayes-best', dest='run_bayes_best', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis using bayes with priors extracted from Monte Carlo')

args = parser.parse_args()
if args.run_stats:
    stats()
if args.run_logLikelihood_by_particle:
    logLikelihood_by_particle()
if args.run_epsilonPID_matrix:
    epsilonPID_matrix()
if args.run_logLikelihood_by_detector:
    logLikelihood_by_detector()
if args.run_mimic_ID:
    mimic_ID()
if args.run_bayes:
    bayes()
if args.run_bayes_best:
    best_priors = {}
    for p in particles:
        best_priors[p] = data[p][data[p]['isSignal'] == 1].size

    bayes(best_priors)
