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
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group_action = parser.add_argument_group('actions', 'Parameters which induce some kind of calculations and or visualizations')
group_opt = parser.add_argument_group('sub-options', 'Parameters which make only sense to use in combination with an action and which possibly alters their behavior')
group_action.add_argument('--stats', dest='run_stats', action='store_true', default=False,
                    help='Print out and visualize some statistics')
group_action.add_argument('--logLikelihood-by-particle', dest='run_logLikelihood_by_particle', action='store_true', default=False,
                    help='Plot the binned logLikelihood for each particle')
group_action.add_argument('--epsilonPID-matrix', dest='run_epsilonPID_matrix', action='store_true', default=False,
                    help='Plot the confusion matrix of every events')
group_action.add_argument('--logLikelihood-by-detector', dest='run_logLikelihood_by_detector', action='store_true', default=False,
                    help='Plot the binned logLikelihood for each detector')
group_action.add_argument('--mimic-id', dest='run_mimic_id', action='store_true', default=False,
                    help='Mimic the calculation of the particle IDs using likelihoods')
group_action.add_argument('--bayes', dest='run_bayes', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis using Bayes')
group_action.add_argument('--bayes-best', dest='run_bayes_best', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis using Bayes with priors extracted from Monte Carlo')
group_action.add_argument('--diff', dest='diff_methods', nargs='?', type=str, action='store', default='', const='id,simple_bayes',
                    help='Compare two given methods of selecting particles; Possible values include id, flat_bayes, simple_bayes, chunked_bayes, chunked_bayes_by_${ROOT_VAR_NAME}')
group_action.add_argument('--chunked-bayes', dest='run_chunked_bayes', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis keeping one variable fixed')
group_action.add_argument('--chunked-bayes-priors', dest='run_chunked_bayes_priors', action='store_true', default=False,
                    help='Visualize the evolution of priors for the chunked Bayesian approach')
group_action.add_argument('--chunked-outliers', dest='run_chunked_outliers', action='store_true', default=False,
                    help='Visualize the outliers of the chunked Bayesian approach')
group_opt.add_argument('--cut', dest='cut', nargs='?', action='store', type=float, default=0.2,
                    help='Position of the default cut if only one is to be performed')
group_opt.add_argument('--hold', dest='hold', nargs='?', action='store', default='pt',
                    help='Variable upon which the a priori probabilities shall depend on')
group_opt.add_argument('--norm', dest='norm', nargs='?', action='store', default='pi+',
                    help='Particle by which to norm the a priori probabilities in the chunked Bayesian approach')
group_opt.add_argument('--nbins', dest='nbins', nargs='?', action='store', type=int, default=10,
                    help='Number of bins to use for splitting the `hold` variable in the chunked Bayesian approach')
group_opt.add_argument('--ncuts', dest='ncuts', nargs='?', action='store', type=int, default=10,
                    help='Number of cuts to perform for the various curves')
group_opt.add_argument('--niterations', dest='niterations', nargs='?', action='store', type=int, default=5,
                    help='Number of iterations to perform for the iterative chunked Bayesian approach')
group_opt.add_argument('--particles-of-interest', dest='particles_of_interest', nargs='?', action='store', default='K+,pi+,mu+',
                    help='List of particles which shall be analysed')
group_opt.add_argument('--whis', dest='whis', nargs='?', action='store', type=float, default=1.5,
                    help='Whiskers with which the IQR will be IQR')

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
variable_units = {'pt': r'$\mathrm{GeV/c}$', 'Theta': r'$Rad$', 'cosTheta': 'unitless'}
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
    for p, particle_data in data.items():
        stat[p] = {'tpr': np.array([]), 'fpr': np.array([]), 'tnr': np.array([]), 'ppv': np.array([])}
        for cut in cuts:
            stat[p]['tpr'] = np.append(stat[p]['tpr'], [np.float64(particle_data[(particle_data['isSignal'] == 1) & (particle_data[cutting_columns[p]] > cut)].shape[0]) / np.float64(particle_data[particle_data['isSignal'] == 1].shape[0])])
            stat[p]['fpr'] = np.append(stat[p]['fpr'], [np.float64(particle_data[(particle_data['isSignal'] == 0) & (particle_data[cutting_columns[p]] > cut)].shape[0]) / np.float64(particle_data[particle_data['isSignal'] == 0].shape[0])])
            stat[p]['tnr'] = np.append(stat[p]['tnr'], [np.float64(particle_data[(particle_data['isSignal'] == 0) & (particle_data[cutting_columns[p]] <= cut)].shape[0]) / np.float64(particle_data[particle_data['isSignal'] == 0].shape[0])])
            stat[p]['ppv'] = np.append(stat[p]['ppv'], [np.float64(particle_data[(particle_data['isSignal'] == 1) & (particle_data[cutting_columns[p]] > cut)].shape[0]) / np.float64(particle_data[particle_data[cutting_columns[p]] > cut].shape[0])])

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

    for particle_data in data.values():
        # Calculate the accumulated logLikelihood and assess the relation to kaonID
        for p in particles:
            particle_data['accumulatedLogLikelihood' + basf2_Code(p)] = np.zeros(particle_data[particleIDs[p]].shape[0])
            # The following loop is equivalent to querying the 'all' pseudo-detector when using flat detector weights
            for d in detectors:
                column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
                # Fill up NaN values for detectors which do not yield a result
                # Since at least one detector must return a logLikelihood it is not possible that only NaN values lead to a probability of 1
                particle_data['accumulatedLogLikelihood' + basf2_Code(p)] += particle_data[column].fillna(0) * detector_weights[d]

        # Calculate the particleIDs manually and compare them to the result of the analysis software
        particle_data['assumed_' + particleIDs['pi+']] = 1. / (1. + (particle_data['accumulatedLogLikelihood' + basf2_Code('K+')] - particle_data['accumulatedLogLikelihood' + basf2_Code('pi+')]).apply(np.exp))
        for p in (set(particles) - set(['pi+'])):
            # Algebraic trick to make exp(a)/(exp(a) + exp(b)) stable even for very small values of a and b
            particle_data['assumed_' + particleIDs[p]] = 1. / (1. + (particle_data['accumulatedLogLikelihood' + basf2_Code('pi+')] - particle_data['accumulatedLogLikelihood' + basf2_Code(p)]).apply(np.exp))

            if check:
                # Assert for equality of the manual calculation and analysis software's output
                npt.assert_allclose(particle_data['assumed_' + particleIDs[p]].values, particle_data[particleIDs[p]].astype(np.float64).values, atol=1e-3)

    print('Successfully calculated the particleIDs using the logLikelihoods only.')


def bayes(priors=defaultdict(lambda: 1., {}), mc_best=False):
    """Compute probabilities for particle hypothesis using a Bayesian approach.

    Args:
        priors: Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.
        mc_best: Boolean specifying whether to use the Monte Carlo data for calculating the a prior probabilities.

    Returns:
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.

    """
    if mc_best == True:
        priors = {p: data[p][data[p]['isSignal'] == 1].shape[0] for p in particles}

    cutting_columns = {k: 'bayes_' + v for k, v in particleIDs.items()}

    for particle_data in data.values():
        # TODO: Use mimic_ID here to allow for weighted detector

        for p in particles:
            denominator = 0.
            for p_2 in particles:
                denominator += (particle_data['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__spall__bc'] - particle_data['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__spall__bc']).apply(np.exp) * priors[p_2]

            # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
            particle_data[cutting_columns[p]] = priors[p] / denominator

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
    for p, particle_data in data.items():
        q75, q25 = np.percentile(particle_data[hold], [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (iqr * whis)
        upper_bound = q75 + (iqr * whis)

        particle_data[category_column], intervals[p] = pd.qcut(particle_data[(particle_data[hold] > lower_bound) & (particle_data[hold] < upper_bound)][hold], q=nbins, labels=range(nbins), retbins=True)

    cutting_columns = {k: 'bayes_' + hold + '_' + v for k, v in particleIDs.items()}
    iteration_priors = {l: {p: [[] for _ in range(niterations)] for p in particles} for l in particles}

    for l, particle_data in data.items():
        for i in range(nbins):
            if mc_best == True:
                y = {p: np.float64(particle_data[(particle_data[category_column] == i) & (particle_data['mcPDG'] == pdg_from_name_faulty(p))].shape[0]) for p in particles}
                priors = {p: y[p] / y[norm] for p in particles}

                print('Priors %d/%d "%s" Bin: '%(i+1, nbins, hold), priors)
            else:
                priors = {p: 1. for p in particles}

            for iteration in range(niterations):
                # Calculate the 'a posteriori' probability for each pt bin
                for p in particles:
                    denominator = 0.
                    for p_2 in particles:
                        denominator += (particle_data[particle_data[category_column] == i]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - particle_data[particle_data[category_column] == i]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

                    # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
                    particle_data.at[particle_data[category_column] == i, cutting_columns[p]] = priors[p] / denominator

                y = {p: np.float64(particle_data[particle_data[category_column] == i][cutting_columns[p]].sum()) for p in particles}
                for p in particles:
                    priors[p] = y[p] / y[norm]
                    iteration_priors[l][p][iteration] += [priors[p]]

                if not mc_best: print('Priors %d/%d "%s" Bin after Iteration %2d: '%(i+1, nbins, hold, iteration + 1), priors)

        max_columns = particle_data[list(cutting_columns.values())].idxmax(axis=1)
        cutting_columns_isMax = {k: v + '_isMax' for k, v in cutting_columns.items()}
        for p in cutting_columns.keys():
            particle_data[cutting_columns_isMax[p]] = np.where(max_columns == cutting_columns[p], 1, 0)

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

    fig, _ = plt.subplots(nrows=len(epsilonPIDs_approaches), ncols=1)
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
        for n in range(1, len(stats_approaches)):
            interpolated_rate = scipy.interpolate.interp1d(stats_approaches[n][p]['fpr'], stats_approaches[n][p]['tpr'], bounds_error=False, fill_value='extrapolate')
            plt.plot(stats_approaches[0][p]['fpr'], interpolated_rate(stats_approaches[0][p]['fpr']) / stats_approaches[0][p]['tpr'], label='TPR%s /%s'%(title_suffixes[n], title_suffixes[0]), color='C2')
            interpolated_rate = scipy.interpolate.interp1d(stats_approaches[n][p]['fpr'], stats_approaches[n][p]['ppv'], bounds_error=False, fill_value='extrapolate')
            plt.plot(stats_approaches[n][p]['fpr'], interpolated_rate(stats_approaches[0][p]['fpr']) / stats_approaches[0][p]['ppv'], label='PPV%s /%s'%(title_suffixes[n], title_suffixes[0]), linestyle=':', color='C3')

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
    cut = args.cut
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
    c = bayes(mc_best=True)
    plot_stats_by_particle(stats(cutting_columns=c))

if args.diff_methods:
    methods = args.diff_methods.split(',')
    assert (len(methods) >= 2), 'Specify at least two methods'

    cut = args.cut
    ncuts = args.ncuts

    hold = args.hold
    whis = args.whis
    nbins = args.nbins
    niterations = args.niterations
    norm = args.norm
    mc_best = False

    particles_of_interest = args.particles_of_interest.split(',')

    epsilonPIDs_approaches = []
    stats_approaches = []
    title_suffixes = []
    for m in methods:
        if m == 'id':
            c = particleIDs
            title_suffixes += [' via PID']
        elif m == 'flat_bayes':
            c = bayes()
            title_suffixes += [' via flat Bayes']
        elif m == 'simple_bayes':
            c = bayes(mc_best=True)
            title_suffixes += [' via simple Bayes']
        elif m == 'chunked_bayes':
            c = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            title_suffixes += [' via chunked Bayes']
        elif re.match(r'chunked_bayes_by_[\w]+', m):
            explicit_hold = re.sub(r'chunked_bayes_by_([\w\d_]+)', r'\1', m)
            c = chunked_bayes(hold=explicit_hold, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            title_suffixes += [' by ' + variable_formats[explicit_hold]]
        else:
            raise ValueError('received unknown method "%s"'%(m))

        epsilonPIDs_approaches += [epsilonPID_matrix(cutting_columns=c, cut=cut)]
        stats_approaches += [stats(cutting_columns=c, ncuts=ncuts)]

    title_epsilonPIDs = r'Heatmap of $\epsilon_{PID}$ matrix for a cut at $%.2f$'%(cut)
    plot_diff_epsilonPIDs(epsilonPIDs_approaches=epsilonPIDs_approaches, title_suffixes=title_suffixes, title_epsilonPIDs=title_epsilonPIDs)
    plot_diff_stats(stats_approaches=stats_approaches, title_suffixes=title_suffixes, particles_of_interest=particles_of_interest)

if args.run_chunked_bayes:
    cut = args.cut

    hold = args.hold
    whis = args.whis
    niterations = args.niterations
    nbins = args.nbins
    norm = args.norm
    cutting_columns, _, category_column, intervals, _ = chunked_bayes(hold=hold, whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals.items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals.items()}

    particles_of_interest = args.particles_of_interest.split(',')

    plt.figure()
    drawing_title = plt.title('True Positive Rate for a Cut at %.2f'%(cut))
    for p in particles_of_interest:
        assumed_abundance = np.array([data[p][(data[p][category_column] == it) & (data[p][cutting_columns[p]] > cut) & (data[p]['isSignal'] == 1)].shape[0] for it in range(nbins)])
        actual_abundance = np.array([data[p][(data[p][category_column] == it) & (data[p]['isSignal'] == 1)].shape[0] for it in range(nbins)])
        plt.errorbar(interval_centers[p], assumed_abundance / actual_abundance, xerr=interval_widths[p], label='%s'%(particle_formats[p]), fmt='o')

    plt.xlabel(variable_formats[hold] + ' (' + variable_units[hold] + ')')
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
    particles_of_interest = args.particles_of_interest.split(',')
    hold = args.hold
    whis = args.whis
    nbins = args.nbins
    niterations = args.niterations
    norm = args.norm

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

        plt.xlabel(variable_formats[hold] + ' (' + variable_units[hold] + ')')
        plt.ylabel('Relative Abundance')
        plt.legend()
        plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Chunked Bayesian Approach: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
        plt.show(block=False)

if args.run_chunked_outliers:
    hold = args.hold
    whis = args.whis
    norm = args.norm

    plt.figure()
    plt.boxplot(data[norm][hold], whis=whis, sym='+')
    drawing_title = plt.title('Outliers Outside of ' + str(whis) + ' IQR on a Logarithmic Scale')
    plt.yscale('log')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel(variable_formats[hold] + ' (' + variable_units[hold] + ')')
    plt.savefig(re.sub('[\\\\$_^{}]', '', 'doc/updates/res/Chunked Bayesian Approach: ' + drawing_title.get_text() + '.pdf'), bbox_inches='tight')
    plt.show(block=False)


plt.show()
