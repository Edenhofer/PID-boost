#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import root_pandas as rpd
import scipy
import scipy.interpolate
import scipy.stats

import pdg


try:
    import seaborn as sns

    # Enable and customize default plotting style
    sns.set_style("whitegrid")
except ImportError:
    pass

try:
    import argcomplete
except ImportError:
    pass


# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.')
parser.add_argument('--stats', dest='run_stats', action='store_true', default=False, help='Print out and visualize some statistics (default: False)')
parser.add_argument('--logLikelihood-by-particle', dest='run_logLikelihood_by_particle', action='store_true', default=False, help='Plot the binned logLikelihood for each particle (default: False)')
parser.add_argument('--epsilonPID-matrix', dest='run_epsilonPID_matrix', action='store_true', default=False, help='Plot the confusion matrix of every events (default: False)')
parser.add_argument('--logLikelihood-by-detector', dest='run_logLikelihood_by_detector', action='store_true', default=False, help='Plot the binned logLikelihood for each detector (default: False)')
parser.add_argument('--mimic-id', dest='run_mimic_id', action='store_true', default=False, help='Mimic the calculation of the particle IDs using likelihoods (default: False)')
parser.add_argument('--bayes', dest='run_bayes', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis using Bayes')
parser.add_argument('--bayes-best', dest='run_bayes_best', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis using Bayes with priors extracted from Monte Carlo')
parser.add_argument('--diff',
                    dest='diff_methods',
                    nargs='?',
                    type=str,
                    action='store',
                    default='',
                    const='id,simple_bayes',
                    help='Compare two given methods of selecting particles (default: id,simple_bayes); Possible values include id, simple_bayes, chunked_bayes')
parser.add_argument('--diff-pt-theta', dest='run_diff_pt_theta', action='store_true', default=False, help='Compare the difference of selecting by particle ID and by chunked Bayes')
parser.add_argument('--chunked-bayes', dest='run_chunked_bayes', action='store_true', default=False, help='Calculate an accumulated probability for particle hypothesis keeping one variable fixed')
parser.add_argument('--chunked-bayes-priors', dest='run_chunked_bayes_priors', action='store_true', default=False, help='Visualize the evolution of priors for the chunked Bayesian approach')
parser.add_argument('--chunked-outliers', dest='run_chunked_outliers', action='store_true', default=False, help='Visualize the outliers of the chunked Bayesian approach')

try:
    argcomplete.autocomplete(parser)
except NameError:
    pass

# Base definitions of stable particles and detector data
particles = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleIDs = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
particle_formats = {'K+': r'$K^+$', 'pi+': r'$\pi^+$', 'e+': r'$e^+$', 'mu+': r'$\mu^+$', 'p+': r'$p^+$', 'deuteron': r'$d$'}
detectors = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']
pseudo_detectors = ['all', 'default']
variable_formats = {'pt': r'$p_T$', 'Theta': r'$\theta$', 'cosTheta': r'$\cos(\theta)$'}
variable_units = {'pt': r'$\mathrm{GeV/c}$', 'Theta': r'$Rad$', 'cosTheta': ''}
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
        raise ValueError('something unexpected happened while converting the input to an escaped pdgCode')


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

    print("epsilon_PID matrix:\n%s"%(epsilonPIDs))
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


def bayes(priors=defaultdict(lambda: 1., {})):
    """Compute probabilities for particle hypothesis using a Bayesian approach.

    Args:
        priors: Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.

    Returns:
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.

    """
    cutting_columns = {k: 'bayes_' + v for k, v in particleIDs.items()}

    for l in particles:
        # TODO: Use mimic_ID here to allow for weighted detector

        for p in particles:
            # Calculate the particleIDs manually and compare them to the result of the analysis software
            denominator = 0.
            for p_2 in particles:
                denominator += (data[l]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__spall__bc'] - data[l]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__spall__bc']).apply(np.exp) * priors[p_2]

            # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
            data[l][cutting_columns[p]] = priors[p] / denominator

    return cutting_columns


def chunked_bayes(hold='pt', nbins=10, detector='all', mc_best=False, niterations=7, norm='pi+', whis=1.5):
    """Compute probabilities for particle hypothesis keeping the `hold` root variable fixed using a Bayesian approach.

    Args:
        hold: Root variable on which the 'a prior' probability shall depend on.
        nbins: Number of bins to use for the `hold` variable when calculating probabilities.
        detector: Name of the detector to be used for pidLogLikelihood extraction.
        mc_best: Boolean specifying whether to use the Monte Carlo data for prior probabilities or an iterative approach.
        niterations: Number of iterations for the converging approach.
        norm: Particle by which abundance to norm the a priori probabilities.
        whis: Whiskers, scale of the Inter Quartile Range (IQR) for outlier exclusion.

    Returns:
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.
        cutting_columns_isMax: A dictionary containing the name of each column by particle which shall be used for exclusive cuts, yielding the maximum probability particle.
        category_column: Name of the column in each dataframe which holds the category for bin selection.
        intervals: A dictionary containing an array of interval boundaries for every bin.
        iteration_priors: A dictionary for each dataset of particle dictionaries containing arrays of priors for each iteration.

    """
    if mc_best == True:
        niterations = 1

    category_column = 'category_' + hold
    intervals = {}
    for p in particles:
        q75, q25 = np.percentile(data[p][hold], [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (iqr * whis)
        upper_bound = q75 + (iqr * whis)

        data[p][category_column], intervals[p] = pd.qcut(data[p][(data[p][hold] > lower_bound) & (data[p][hold] < upper_bound)][hold], q=nbins, labels=range(nbins), retbins=True)

    cutting_columns = {k: 'bayes_' + hold + '_' + v for k, v in particleIDs.items()}
    iteration_priors = {l: {p: [[] for _ in range(niterations)] for p in particles} for l in particles}

    for l in particles:
        for i in range(nbins):
            if mc_best == True:
                y = {p: np.float64(data[l][(data[l][category_column] == i) & (data[l]['mcPDG'] == pdg_from_name_faulty(p))].shape[0]) for p in particles}
                priors = {p: y[p] / y[norm] for p in particles}

                print('Priors %d/%d "%s" Bin: '%(i+1, nbins, hold), priors)
            else:
                priors = {p: 1. for p in particles}

            for iteration in range(niterations):
                # Calculate the 'a posteriori' probability for each pt bin
                for p in particles:
                    denominator = 0.
                    for p_2 in particles:
                        denominator += (data[l][data[l][category_column] == i]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - data[l][data[l][category_column] == i]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

                    # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
                    data[l].at[data[l][category_column] == i, cutting_columns[p]] = priors[p] / denominator

                y = {p: np.float64(data[l][data[l][category_column] == i][cutting_columns[p]].sum()) for p in particles}
                for p in particles:
                    priors[p] = y[p] / y[norm]
                    iteration_priors[l][p][iteration] += [priors[p]]

                if not mc_best: print('Priors %d/%d "%s" Bin after Iteration %2d: '%(i+1, nbins, hold, iteration + 1), priors)

        max_columns = data[l][list(cutting_columns.values())].idxmax(axis=1)
        cutting_columns_isMax = {k: v + '_isMax' for k, v in cutting_columns.items()}
        for p in cutting_columns.keys():
            data[l][cutting_columns_isMax[p]] = np.where(max_columns == cutting_columns[p], 1, 0)

    return cutting_columns, cutting_columns_isMax, category_column, intervals, iteration_priors


def plot_logLikelihood_by_particle(nbins=50):
    for d in detectors + pseudo_detectors:
        plt.figure()
        drawing_title = plt.suptitle('Binned pidLogLikelihood for Detector %s'%(d))
        for i, p in enumerate(particles):
            for i_2, p_2 in enumerate(particles):
                plt.subplot(len(particles), len(particles), i*len(particles)+i_2+1)
                plt.title('Identified %s as %s'%(particle_formats[p], particle_formats[p_2]))
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + d + '__bc'
                data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/logLikelihood by Particle: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)


def plot_logLikelihood_by_detector(nbins=50):
    for p in particles:
        plt.figure()
        drawing_title = plt.suptitle('Binned pidLogLikelihood for Particle %s'%(particle_formats[p]))
        for i, d in enumerate(detectors + pseudo_detectors):
            plt.subplot(2, len(detectors + pseudo_detectors), i+1)
            plt.title('Detector %s with Signal'%(d))
            column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
            data[p][data[p]['isSignal'] == 1][column].hist(bins=nbins)

            plt.subplot(2, len(detectors + pseudo_detectors), i+1+len(detectors + pseudo_detectors))
            plt.title('Detector %s with no Signal'%(d))
            column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
            data[p][data[p]['isSignal'] == 0][column].hist(bins=nbins)

        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/logLikelihood by Detector: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)


def plot_stats_by_particle(stat):
    for p in particles:
        plt.figure()
        plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='ROC')
        # Due to the fact that FPR + TNR = 1 the plot will simply show a straight line; Use for debugging only
        # plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
        plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='PPV')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        drawing_title = plt.title('%s Identification'%(particle_formats[p]))
        plt.legend()
        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Statistics: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)


def plot_diff_epsilonPIDs(epsilonPIDs_approaches=[], title_suffixes=[], title_epsilonPIDs=''):
    if len(epsilonPIDs_approaches) >= 0 and len(epsilonPIDs_approaches) != len(title_suffixes):
        raise ValueError('epsilonPIDs_approaches array must be of same length as the title_suffixes array')

    fig, _ = plt.subplots(nrows=2, ncols=1)
    drawing_title = plt.suptitle(title_epsilonPIDs)
    for n in range(len(epsilonPIDs_approaches)):
        plt.subplot(1, len(epsilonPIDs_approaches), n+1)
        plt.imshow(epsilonPIDs_approaches[n], cmap='viridis')
        for (j, i), label in np.ndenumerate(epsilonPIDs_approaches[n]):
            plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
        plt.grid(b=False, axis='both')
        plt.xlabel('Predicted Particle')
        plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
        plt.ylabel('True Particle')
        plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
        plt.title('ID' + title_suffixes[n])
        plt.tight_layout(pad=1.4)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.20, 0.05, 0.6])
    plt.colorbar(cax=cbar_ax)
    plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Diff Heatmap: ' + drawing_title.get_text() + ','.join(str(suffix) for suffix in title_suffixes) + '.pdf'), bbox_inches='tight')
    plt.show(block=False)


def plot_diff_stats(stats_approaches=[], title_suffixes=[], particles_of_interest=particles):
    if len(stats_approaches) >= 0 and len(stats_approaches) != len(title_suffixes):
        raise ValueError('stats_approaches array must be of same length as the title_suffixes array')

    for p in particles_of_interest:
        plt.figure()
        grid = plt.GridSpec(3, 1, hspace=0.1)

        main_ax = plt.subplot(grid[:2, 0])
        drawing_title = plt.title('%s Identification'%(particle_formats[p]))
        for n in range(len(stats_approaches)):
            drawing = plt.plot(stats_approaches[n][p]['fpr'], stats_approaches[n][p]['tpr'], label='ROC' + title_suffixes[n])
            plt.plot(stats_approaches[n][p]['fpr'], stats_approaches[n][p]['ppv'], label='PPV' + title_suffixes[n], linestyle=':', color=drawing[0].get_color())

        plt.setp(main_ax.get_xticklabels(), visible=False)
        plt.ylabel('Particle Rates')
        plt.legend()

        plt.subplot(grid[2, 0], sharex=main_ax)
        for n in range(0, len(stats_approaches), 2):
            interpolated_rate = scipy.interpolate.interp1d(stats_approaches[n+1][p]['fpr'], stats_approaches[n+1][p]['tpr'], bounds_error=False, fill_value='extrapolate')
            plt.plot(stats_approaches[n][p]['fpr'], interpolated_rate(stats_approaches[n][p]['fpr']) / stats_approaches[n][p]['tpr'], label='TPR Ratio', color='C2')
            interpolated_rate = scipy.interpolate.interp1d(stats_approaches[n+1][p]['fpr'], stats_approaches[n+1][p]['ppv'], bounds_error=False, fill_value='extrapolate')
            plt.plot(stats_approaches[n][p]['fpr'], interpolated_rate(stats_approaches[n][p]['fpr']) / stats_approaches[n][p]['ppv'], label='PPV Ratio', linestyle=':', color='C3')

        plt.axhline(y=1., color='dimgrey', linestyle='--')
        plt.grid(b=True, axis='both')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Rate Ratios')
        plt.legend()

        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Diff Statistics: ' + drawing_title.get_text() + ','.join(str(suffix) for suffix in title_suffixes) + '.pdf'), bbox_inches='tight')
        plt.show(block=False)


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

    plt.figure()
    plt.imshow(epsilonPIDs, cmap='viridis')
    for (j, i), label in np.ndenumerate(epsilonPIDs):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
    plt.grid(b=False, axis='both')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
    plt.colorbar()
    drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut))
    plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
    plt.show(block=False)

if args.run_mimic_id:
    mimic_ID()

if args.run_bayes:
    c = bayes()
    plot_stats_by_particle(stats(cutting_columns=c))

if args.run_bayes_best:
    best_priors = {p: data[p][data[p]['isSignal'] == 1].shape[0] for p in particles}

    c = bayes(best_priors)
    plot_stats_by_particle(stats(cutting_columns=c))

if args.diff_methods:
    methods = args.diff_methods.split(',')
    assert(len(methods) == 2) # Currently only allow two different methods to be compared

    cut = 0.2
    ncuts = 10

    hold = 'pt'
    whis = 1.5
    nbins = 10
    niterations = 5
    norm = 'pi+'
    mc_best = False

    particles_of_interest = ['K+', 'pi+', 'mu+']

    if set(methods) == {'id', 'simple_bayes'}:
        cutting_columns_first = particleIDs
        best_priors = {p: data[p][data[p]['isSignal'] == 1].shape[0] for p in particles}
        cutting_columns_second = bayes(best_priors)

        title_suffixes = [' via ID', ' via Priors']

    if set(methods) == {'id', 'chunked_bayes'}:
        cutting_columns_first = particleIDs
        cutting_columns_second = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]

        title_suffixes = [' via ID', ' via chunked Bayes']

    if set(methods) == {'simple_bayes', 'chunked_bayes'}:
        cutting_columns_second = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
        best_priors = {p: data[p][data[p]['isSignal'] == 1].shape[0] for p in particles}
        cutting_columns_first = bayes(best_priors)

        title_suffixes = [' via simple Bayes', ' via chunked Bayes']

    epsilonPIDs_approaches = [epsilonPID_matrix(cutting_columns=cutting_columns_first, cut=cut), epsilonPID_matrix(cutting_columns=cutting_columns_second, cut=cut)]
    stats_approaches = [stats(cutting_columns=cutting_columns_first, ncuts=ncuts), stats(cutting_columns=cutting_columns_second, ncuts=ncuts)]

    title_epsilonPIDs = r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut)
    plot_diff_epsilonPIDs(epsilonPIDs_approaches=epsilonPIDs_approaches, title_suffixes=title_suffixes, title_epsilonPIDs=title_epsilonPIDs)
    plot_diff_stats(stats_approaches=stats_approaches, title_suffixes=title_suffixes, particles_of_interest=particles_of_interest)

if args.run_diff_pt_theta:
    cut = 0.2
    ncuts = 10

    hold1 = 'pt'
    hold2 = 'cosTheta'
    whis = 1.5
    nbins = 10
    niterations = 5
    norm = 'pi+'
    cutting_columns_by_pt = chunked_bayes(hold=hold1, whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)[0]
    cutting_columns_by_theta = chunked_bayes(hold=hold2, whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)[0]

    stat_by_pt = stats(cutting_columns=cutting_columns_by_pt, ncuts=ncuts)
    stat_by_theta = stats(cutting_columns=cutting_columns_by_theta, ncuts=ncuts)

    epsilonPIDs_by_pt = epsilonPID_matrix(cutting_columns=cutting_columns_by_pt, cut=cut)
    epsilonPIDs_by_theta = epsilonPID_matrix(cutting_columns=cutting_columns_by_theta, cut=cut)

    plot_diff_epsilonPIDs(epsilonPIDs_approaches=[epsilonPIDs_by_pt, epsilonPIDs_by_theta], title_suffixes=[' by ' + variable_formats[hold1], ' by ' + variable_formats[hold2]], title_epsilonPIDs=r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut))
    plot_diff_stats(stats_approaches=[stat_by_pt, stat_by_theta], title_suffixes=[' by ' + variable_formats[hold1], ' by ' + variable_formats[hold2]], particles_of_interest=['K+', 'pi+', 'mu+'])

if args.run_chunked_bayes:
    particle_visuals = {'K+': 'C0', 'pi+': 'C1'}
    cuts = [0.2]

    hold = 'pt'
    whis = 1.5
    nbins = 10
    niterations = 5
    norm = 'pi+'
    cutting_columns, _, category_column, intervals, _ = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals.items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals.items()}

    for cut in cuts:
        plt.figure()
        drawing_title = plt.title('True Positive Rate for a Cut at %.2f'%(cut))
        for p, color in particle_visuals.items():
            assumed_abundance = np.array([data[p][(data[p][category_column] == it) & (data[p][cutting_columns[p]] > cut) & (data[p]['isSignal'] == 1)].shape[0] for it in range(nbins)])
            actual_abundance = np.array([data[p][(data[p][category_column] == it) & (data[p]['isSignal'] == 1)].shape[0] for it in range(nbins)])
            plt.errorbar(interval_centers[p], assumed_abundance / actual_abundance, xerr=interval_widths[p], label='%s'%(particle_formats[p]), fmt='o', color=color)

        plt.xlabel('Transverse Momentum ' + variable_formats[hold] + ' (' + variable_units[hold] + ')')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Chunked Bayesian Approach: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)

        epsilonPIDs = epsilonPID_matrix(cutting_columns=cutting_columns, cut=cut)
        plt.figure()
        plt.imshow(epsilonPIDs, cmap='viridis')
        for (j, i), label in np.ndenumerate(epsilonPIDs):
            plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
        plt.grid(b=False, axis='both')
        plt.xlabel('Predicted Particle')
        plt.xticks(range(len(particles)), [particle_formats[p] for p in particles])
        plt.ylabel('True Particle')
        plt.yticks(range(len(particles)), [particle_formats[p] for p in particles])
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut))
        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Chunked Bayesian Approach: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)

    plot_stats_by_particle(stats(cutting_columns=cutting_columns))

if args.run_chunked_bayes_priors:
    particles_of_interest = ['K+', 'mu+']

    hold = 'pt'
    whis = 1.5
    nbins = 10
    niterations = 3
    norm = 'pi+'

    iteration_priors_viaIter = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)[-1]
    intervals, iteration_priors_viaBest = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=True, nbins=nbins)[-2]
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals.items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals.items()}

    for p in particles_of_interest:
        plt.figure()
        drawing_title = plt.title('%s Spectra Ratios Relative to %s'%(particle_formats[p], particle_formats[norm]))
        plt.errorbar(interval_centers[p], iteration_priors_viaBest[norm][p][-1], xerr=interval_widths[p], label='Truth', fmt='*')
        for n in range(niterations):
            plt.errorbar(interval_centers[p], iteration_priors_viaIter[norm][p][n], xerr=interval_widths[p], label='Iteration %d'%(n+1), fmt='o')

        plt.xlabel('Transverse Momentum ' + variable_formats[hold] + ' (' + variable_units[hold] + ')')
        plt.ylabel('Relative Abundance')
        plt.legend()
        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Chunked Bayesian Approach: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)

if args.run_chunked_outliers:
    hold = 'pt'
    whis = 1.5
    norm = 'pi+'

    plt.figure()
    plt.boxplot(data[norm][hold], whis=whis, sym='+')
    drawing_title = plt.title('Outliers Outside of ' + str(whis) + ' IQR on a Logarithmic Scale')
    plt.yscale('log')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel('Transverse Momentum ' + variable_formats[hold] + ' (' + variable_units[hold] + ')')
    plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Chunked Bayesian Approach: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
    plt.show(block=False)


plt.show()
