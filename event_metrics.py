#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argcomplete
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import root_pandas as rpd
import scipy
import scipy.stats

import pdg


# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.')
parser.add_argument('--stats', dest='run_stats', action='store_true', default=False, help='Print out and visualize some statistics (default: False)')
parser.add_argument('--logLikelihood-by-particle', dest='run_logLikelihood_by_particle', action='store_true', default=False, help='Plot the binned logLikelihood for each particle (default: False)')
parser.add_argument('--epsilonPID-matrix', dest='run_epsilonPID_matrix', action='store_true', default=False, help='Plot the confusion matrix of every events (default: False)')
parser.add_argument('--logLikelihood-by-detector', dest='run_logLikelihood_by_detector', action='store_true', default=False, help='Plot the binned logLikelihood for each detector (default: False)')
parser.add_argument('--mimic-ID', dest='run_mimic_ID', action='store_true', default=False, help='Mimic the calculation of the particle IDs using likelihoods (default: False)')
parser.add_argument('--bayes', dest='run_bayes', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis using bayes')
parser.add_argument('--bayes-best', dest='run_bayes_best', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis using bayes with priors extracted from Monte Carlo')
parser.add_argument('--diff-ID-Bayes', dest='run_diff_ID_Bayes', action='store_true', default=False, help='Compare the difference of selecting by particle ID and bayes')
parser.add_argument('--chunked-bayes', dest='run_chunked_bayes', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis keeping one variable fixed')
parser.add_argument('--chunked-outliers', dest='run_chunked_outliers', action='store_true', default=False, help='Visualize the outliers of the chunked Bayesian approach')
argcomplete.autocomplete(parser)

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
data = {p: rpd.read_root(p + '.root') for p in particles}


def pdg_from_name_faulty(particle):
    """Return the pdgCode for a given particle honoring a bug in the float to integer conversion.

    Args:
        particle: The name of the particle which should be translated.

    Returns:
        The Particle Data Group (PDG) code compatible with the values in root files.

    """
    if particle == 'deuteron':
        return 1000010048
    if particle == 'anti-deuteron':
        return -1000010048
    else:
        return pdg.from_name(particle)


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
        stat[p] = {'tpr': np.array([]), 'fpr': np.array([]), 'tnr': np.array([]), 'ppv': np.array([])}
        for cut in cuts:
            stat[p]['tpr'] = np.append(stat[p]['tpr'], [np.float64(data[p][(data[p]['isSignal'] == 1) & (data[p][cutting_columns[p]] > cut)].shape[0]) / np.float64(data[p][data[p]['isSignal'] == 1].shape[0])])
            stat[p]['fpr'] = np.append(stat[p]['fpr'], [np.float64(data[p][(data[p]['isSignal'] == 0) & (data[p][cutting_columns[p]] > cut)].shape[0]) / np.float64(data[p][data[p]['isSignal'] == 0].shape[0])])
            stat[p]['tnr'] = np.append(stat[p]['tnr'], [np.float64(data[p][(data[p]['isSignal'] == 0) & (data[p][cutting_columns[p]] <= cut)].shape[0]) / np.float64(data[p][data[p]['isSignal'] == 0].shape[0])])
            stat[p]['ppv'] = np.append(stat[p]['ppv'], [np.float64(data[p][(data[p]['isSignal'] == 1) & (data[p][cutting_columns[p]] > cut)].shape[0]) / np.float64(data[p][data[p][cutting_columns[p]] > cut].shape[0])])

            if not np.isclose(stat[p]['fpr'][-1]+stat[p]['tnr'][-1], 1, atol=1e-2):
                print('VALUES INCONSISTENT: ', end='')

            print('Particle %10s: TPR=%6.6f; FPR=%6.6f; TNR=%6.6f; PPV=%6.6f; cut=%4.4f'%(p, stat[p]['tpr'][-1], stat[p]['fpr'][-1], stat[p]['tnr'][-1], stat[p]['ppv'][-1], cut))

    return stat


def epsilonPID_matrix(cut=0.2, cutting_columns=particleIDs):
    """Calculate the epsilon_PID matrix for misclassifying particles, print the result and plot a heatmap.

    Args:
        cut: Position of the cut for the cutting_columns.
        cutting_columns: Dictionary which yields a column name for each particle on which the cuts are performed.

    Returns:
        A numpy matrix of epsilon_PID values. The `epsilon_PID[i][j]` value being the probability given it is a particle ''i'' that it will be categorized as particle ''j''.

    """
    epsilonPIDs = np.zeros(shape=(len(particles), len(particles)))
    for i, i_p in enumerate(particles):
        for j, j_p in enumerate(particles):
            # The deuterium code is not properly stored in the mcPDG variable, hence the use of `pdg_from_name_faulty()`
            epsilonPIDs[i][j] = np.float64(data[i_p][(data[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) & (data[i_p][cutting_columns[j_p]] > cut)].shape[0]) / np.float64(data[i_p][data[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)].shape[0])

    print("Confusion matrix:\n%s"%(epsilonPIDs))
    return epsilonPIDs


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
            data[l]['accumulatedLogLikelihood' + basf2_Code(p)] = np.zeros(data[l][particleIDs[p]].shape[0])
            # The following loop is equivalent to querying the 'all' pseudo-detector when using flat detector weights
            for d in detectors:
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
                # Fill up NaN values for detectors which do not yield a result
                # Since at least one detector must return a logLikelihood it is not possible that only NaN values lead to a probability of 1
                data[l]['accumulatedLogLikelihood' + basf2_Code(p)] += data[l][column].fillna(0) * detector_weights[d]

        # Calculate the particleIDs manually and compare them to the result of the analysis software
        data[l]['assumed_' + particleIDs['pi+']] = 1. / (1. + (data[l]['accumulatedLogLikelihood' + basf2_Code('K+')] - data[l]['accumulatedLogLikelihood' + basf2_Code('pi+')]).apply(np.exp))
        for p in (set(particles) - set(['pi+'])):
            # Algebraic trick to make exp(a)/(exp(a) + exp(b)) stable even for very small values of a and b
            data[l]['assumed_' + particleIDs[p]] = 1. / (1. + (data[l]['accumulatedLogLikelihood' + basf2_Code('pi+')] - data[l]['accumulatedLogLikelihood' + basf2_Code(p)]).apply(np.exp))

            if check:
                # Assert for equality of the manual calculation and analysis software's output
                npt.assert_allclose(data[l]['assumed_' + particleIDs[p]].values, data[l][particleIDs[p]].astype(np.float64).values, atol=1e-3)

    print('Successfully calculated the particleIDs using the logLikelihoods only.')


def bayes(priors=defaultdict(lambda: 1., {}), **kwargs):
    """Compute probabilities for particle hypothesis using a bayesian approach.

    Args:
        priors: Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.

        Any keyword arguments which are accepted by the `stats` function.

    Returns:
        stat: The output of the `stats` function for cutting at the newly added bayesian particle ID. See the `stats` function return value.
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.

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

    cutting_columns = {k: 'bayes_' + v for k, v in particleIDs.items()}
    stat = stats(cutting_columns=cutting_columns, **kwargs)

    return stat, cutting_columns


def chunked_bayes(hold='pt', nbins=10, detector='all', mc_best=False, niterations=7, norm='pi+'):
    """Compute probabilities for particle hypothesis keeping the `hold` root variable fixed using a bayesian approach.

    Args:
        hold: Root variable on which the 'a prior' probability shall depend on.
        nbins: Number of bins to use for the `hold` variable when calculating probabilities.
        detector: Name of the detector to be used for pidLogLikelihood extraction.
        mc_best: Boolean specifying whether to use the Monte Carlo data for prior probabilities or an iterative approach.
        niterations: Number of iterations for the converging approach.
        norm: Particle by which abundance to norm the a priori probabilities.

    Returns:
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.
        bins: A dictionary containing a list for each particle dataset with the category of the `hold` variable bins of each track.
        intervals: A dictionary containing an array of interval boundaries for every bin.

    """
    bins = {}
    intervals = {}
    for p in particles:
        bins[p], intervals[p] = pd.qcut(data[p][hold], q=nbins, labels=range(nbins), retbins=True)

    cutting_columns = {k: 'bayes_' + hold + '_' + v for k, v in particleIDs.items()}

    for l in particles:
        for i in range(nbins):
            if mc_best == True:
                y = {p: np.float64(data[l][(bins[l] == i) & (data[l]['mcPDG'] == pdg_from_name_faulty(p))].shape[0]) for p in particles}
                priors = {p: y[p] / y[norm] for p in particles}
                niterations = 1

                print('Priors %d/%d "%s" bin: '%(i+1, nbins, hold), priors)
            else:
                priors = {p: 1. for p in particles}

            for iteration in range(niterations):
                # Calculate the 'a posteriori' probability for each pt bin
                for p in particles:
                    denominator = 0.
                    for p_2 in particles:
                        denominator += (data[l][bins[l] == i]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - data[l][bins[l] == i]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

                    # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
                    data[l].at[bins[l] == i, cutting_columns[p]] = priors[p] / denominator

                y = {p: np.float64(data[l][bins[l] == i][cutting_columns[p]].sum()) for p in particles}
                for p in particles:
                    priors[p] = y[p] / y[norm]

                if not mc_best: print('Priors %d/%d "%s" bin after iteration %2d: '%(i+1, nbins, hold, iteration + 1), priors)

    return cutting_columns, bins, intervals


def plot_logLikelihood_by_particle(nbins=50):
    for d in detectors + pseudo_detectors:
        plt.suptitle('Binned pidLogLikelihood for detector %s'%(d))
        for i, p in enumerate(particles):
            for i_2, p_2 in enumerate(particles):
                plt.subplot(len(particles), len(particles), i*len(particles)+i_2+1)
                plt.title('Identified %s as %s'%(particle_formats[p], particle_formats[p_2]))
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + d + '__bc'
                data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

        plt.show()


def plot_logLikelihood_by_detector(nbins=50):
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


def plot_stats_by_particle(stat):
    for p in particles:
        plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='True Positive Rate (ROC curve)')
        # Due to the fact that FPR + TNR = 1 the plot will simply show a straight line; Use for debugging only
        # plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
        plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='Positive Predicted Value')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        plt.title('%s identification'%(particle_formats[p]))
        plt.legend()
        plt.show()


args = parser.parse_args()
if args.run_stats:
    plot_stats_by_particle(stats())

if args.run_logLikelihood_by_particle:
    plot_logLikelihood_by_particle()

if args.run_logLikelihood_by_detector:
    plot_logLikelihood_by_detector()

if args.run_epsilonPID_matrix:
    cut = 0.2
    epsilonPIDs = epsilonPID_matrix(cut=cut)

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

if args.run_mimic_ID:
    mimic_ID()

if args.run_bayes:
    stat, c = bayes()
    plot_stats_by_particle(stat)

if args.run_bayes_best:
    best_priors = {p: data[p][data[p]['isSignal'] == 1].shape[0] for p in particles}

    stat, c = bayes(best_priors)
    plot_stats_by_particle(stat)

if args.run_diff_ID_Bayes:
    cut = 0.2
    ncuts = 10

    best_priors = {p: data[p][data[p]['isSignal'] == 1].shape[0] for p in particles}

    stat_viaPrior, c = bayes(best_priors, ncuts=ncuts)
    stat_viaID = stats(ncuts=ncuts)

    epsilonPIDs_viaPrior = epsilonPID_matrix(cutting_columns=c, cut=cut)
    epsilonPIDs_viaID = epsilonPID_matrix(cut=cut)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    plt.suptitle(r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut))

    plt.subplot(1, 2, 1)
    plt.imshow(epsilonPIDs_viaID)
    for (j, i), label in np.ndenumerate(epsilonPIDs_viaID):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.title('Identification via ID')

    plt.subplot(1, 2, 2)
    plt.imshow(epsilonPIDs_viaPrior)
    for (j, i), label in np.ndenumerate(epsilonPIDs_viaPrior):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.title('Identification via Bayes')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(cax=cbar_ax)

    plt.show(fig)

    for p in particles:
        plt.subplot(1, 2, 1)
        plt.title('%s identification via ID'%(particle_formats[p]))
        plt.plot(stat_viaID[p]['fpr'], stat_viaID[p]['tpr'], label='True Positive Rate (ROC curve)')
        plt.plot(stat_viaID[p]['fpr'], stat_viaID[p]['ppv'], label='Positive Predicted Value')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('%s identification via Bayes'%(particle_formats[p]))
        plt.plot(stat_viaPrior[p]['fpr'], stat_viaPrior[p]['tpr'], label='True Positive Rate (ROC curve)')
        plt.plot(stat_viaPrior[p]['fpr'], stat_viaPrior[p]['ppv'], label='Positive Predicted Value')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        plt.legend()

        plt.show()

if args.run_chunked_bayes:
    particle_visuals = {'K+': 'C0', 'pi+': 'C1'}
    cut_visuals = {0.1: '-', 0.3: ':', 0.5: '-.', 0.7: '--'}

    nbins = 10
    niterations = 5
    norm = 'pi+'
    cutting_columns, bins, intervals = chunked_bayes(hold='pt', norm=norm, mc_best=False, niterations=niterations, nbins=nbins)
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals.items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals.items()}

    plt.title('Chunked Bayes Abundance Comparison')

    for cut, linestyle in cut_visuals.items():
        for p, color in particle_visuals.items():
            assumed_abundance = np.array([data[p][(bins[p] == it) & (data[p][cutting_columns[p]] > cut) & (data[p]['isSignal'] == 1)].shape[0] for it in range(nbins)])
            actual_abundance = np.array([data[p][(bins[p] == it) & (data[p]['isSignal'] == 1)].shape[0] for it in range(nbins)])
            plt.errorbar(interval_centers[p], assumed_abundance / actual_abundance, xerr=interval_widths[p], label='%s: %.2f cut'%(particle_formats[p], cut), linestyle=linestyle, color=color)

    plt.xscale('log')
    plt.xlabel(r'$p_T$ bin')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

if args.run_chunked_outliers:
    hold = 'pt'
    hold_format = r'$p_T$'
    hold_unit = r'$\mathrm{GeV/c}$'
    whis = 1.5
    norm = 'pi+'

    plt.boxplot(data[norm][hold], whis=whis, sym='+')
    plt.title('Outliers outside of ' + str(whis) + ' IQR on a logarithmic scale')
    plt.yscale('log')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel('Transverse Momentum ' + hold_format + ' (' + hold_unit + ')')
    plt.show()
