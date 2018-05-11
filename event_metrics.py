#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse
import re
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import root_pandas as rpd
import sys

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


# Base definitions of stable particles and detector data
particles = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
particleIDs = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
particle_formats = {'K+': r'$K^+$', 'K-': r'$K^-$', 'pi+': r'$\pi^+$', 'pi-': r'$\pi^-$', 'e+': r'$e^+$', 'e-': r'$e^-$', 'mu+': r'$\mu^+$', 'mu-': r'$\mu^-$', 'p+': r'$p^+$', 'p-': r'$p^-$', 'anti-p-': r'$\bar{p}^-$', 'deuteron': r'$d$', 'anti-deuteron': r'$\bar{d}$', 'Sigma+': r'$\Sigma^+$', 'Sigma-': r'$\Sigma^-$', 'anti-Sigma+': r'$\bar{\Sigma}^+$', 'anti-Sigma-': r'$\bar{\Sigma}^-$', 'Xi+': r'$\Xi^+$', 'Xi-': r'$\Xi^-$', 'anti-Xi+': r'$\bar{\Xi}^+$', 'anti-Xi-': r'$\bar{\Xi}^-$', 'None': r'$None$', 'nan': r'$NaN$'}
particle_base_formats = {'K+': r'$K$', 'K-': r'$K$', 'pi+': r'$\pi$', 'pi-': r'$\pi$', 'e+': r'$e$', 'e-': r'$e$', 'mu+': r'$\mu$', 'mu-': r'$\mu$', 'p+': r'$p$', 'p-': r'$p$', 'deuteron': r'$d$', 'Sigma+': r'$\Sigma$', 'Sigma-': r'$\Sigma$', 'Xi+': r'$\Xi$', 'Xi-': r'$\Xi$', 'None': r'$None$', 'nan': r'$NaN$'}
detectors = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']
pseudo_detectors = ['all', 'default']
variable_formats = {'p': r'$p$', 'pErr': r'$p_{Err}$', 'phi': r'$\phi$', 'phiErr': r'$\phi_{Err}$', 'pt': r'$p_t$', 'ptErr': r'${p_t}_{Err}$', 'z0': r'$z0$', 'd0': r'$d0$', 'omega': r'$\omega$', 'omegaErr': r'$\omega_{Err}$', 'Theta': r'$\Theta$', 'ThetaErr': r'$\Theta_{Err}$', 'cosTheta': r'$\cos(\Theta)$'}
variable_units = {'p': r'$\mathrm{GeV/c}$', 'phi': r'$Rad$', 'pt': r'$\mathrm{GeV/c}$', 'z0': r'$?$', 'd0': r'$?$', 'omega': r'$?$', 'Theta': r'$Rad$', 'cosTheta': r'$unitless$'}
# Use the detector weights to exclude certain detectors, e.g. for debugging purposes
# Bare in mind that if all likelihoods are calculated correctly this should never improve the result
detector_weights = {d: 1. for d in detectors + pseudo_detectors}

# Dictionary of variables and their boundaries for possible values they might yield
physical_boundaries = {'pt': (0, 5.5), 'cosTheta': (-1, 1)}


# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group_action = parser.add_argument_group('actions', 'Parameters which induce some kind of calculations and or visualizations')
group_opt = parser.add_argument_group('sub-options', 'Parameters which make only sense to use in combination with an action and which possibly alters their behavior')
group_util = parser.add_argument_group('utility options', 'Parameters for altering the behavior of the program\'s input-output handling')
group_action.add_argument('--stats', dest='run_stats', action='store_true', default=False,
                    help='Visualize some general purpose statistics of the dataset')
group_action.add_argument('--pid', dest='run_pid', action='store_true', default=False,
                    help='Print out and visualize some statistics and the epsilonPID-matrix for the default particle ID cut')
group_action.add_argument('--mimic-pid', dest='run_mimic_pid', action='store_true', default=False,
                    help='Mimic the calculation of the particle IDs using likelihoods')
group_action.add_argument('--bayes', dest='run_bayes', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis using Bayes')
group_action.add_argument('--diff', dest='diff_methods', nargs='+', action='store', choices=['pid', 'flat_bayes', 'simple_bayes', 'univariate_bayes', *['univariate_bayes_by_' + v for v in variable_formats.keys()], 'multivariate_bayes'], default=[],
                    help='Compare two given methods of selecting particles')
group_action.add_argument('--univariate-bayes', dest='run_univariate_bayes', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis keeping one variable fixed')
group_action.add_argument('--univariate-bayes-priors', dest='run_univariate_bayes_priors', action='store_true', default=False,
                    help='Visualize the evolution of priors for the univariate Bayesian approach')
group_action.add_argument('--univariate-bayes-outliers', dest='run_univariate_bayes_outliers', action='store_true', default=False,
                    help='Visualize the outliers of the univariate Bayesian approach')
group_action.add_argument('--multivariate-bayes', dest='run_multivariate_bayes', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis keeping multiple variables fixed')
group_action.add_argument('--multivariate-bayes-motivation', dest='run_multivariate_bayes_motivation', action='store_true', default=False,
                    help='Motivate the usage of a multivariate Bayesian approach')
group_opt.add_argument('--cut', dest='cut', nargs='?', action='store', type=float, default=0.2,
                    help='Position of the default cut if only one is to be performed')
group_opt.add_argument('--exclusive-cut', dest='exclusive_cut', action='store_true', default=False,
                    help='Perform exclusive cuts where apropiate using the maximum of the cutting column')
group_opt.add_argument('--hold', dest='hold', nargs='?', action='store', default='pt',
                    help='Variable upon which the a priori probabilities shall depend on')
group_opt.add_argument('--holdings', dest='holdings', nargs='+', action='store', choices=['pt', 'cosTheta'], default=['pt', 'cosTheta'],
                    help='Variables upon which the multivariate a priori probabilities shall depend on')
group_opt.add_argument('--norm', dest='norm', nargs='?', action='store', default='pi+',
                    help='Particle by which to norm the a priori probabilities in the univariate and multivariate Bayesian approach')
group_opt.add_argument('--nbins', dest='nbins', nargs='?', action='store', type=int, default=10,
                    help='Number of bins to use for splitting the `hold` variable in the univariate and multivariate Bayesian approach')
group_opt.add_argument('--ncuts', dest='ncuts', nargs='?', action='store', type=int, default=10,
                    help='Number of cuts to perform for the various curves')
group_opt.add_argument('--niterations', dest='niterations', nargs='?', action='store', type=int, default=5,
                    help='Number of iterations to perform for the iterative univariate and multivariate Bayesian approach')
group_opt.add_argument('--mc-best', dest='mc_best', action='store_true', default=False,
                    help='Use Monte Carlo information for calculating the best possible a priori probabilities instead of relying on an iterative approach')
group_opt.add_argument('--particles-of-interest', dest='particles_of_interest', nargs='+', action='store', choices=particles, default=particles,
                    help='List of particles which shall be analysed')
group_opt.add_argument('--whis', dest='whis', nargs='?', action='store', type=float, default=None,
                    help='Whiskers with which the IQR will be IQR')
group_util.add_argument('-i', '--input', dest='input_directory', action='store', default='./',
                    help='Directory in which the program shall search for root files for each particle')
group_util.add_argument('-o', '--output', dest='output_directory', action='store', default='doc/updates/res/',
                    help='Directory for the generated output (mainly plots); Skip saving plots if given \'/dev/null\'.')
group_util.add_argument('--interactive', dest='interactive', action='store_true', default=True,
                    help='Run interactively, i.e. show plots')
group_util.add_argument('--non-interactive', dest='interactive', action='store_false', default=True,
                    help='Run non-interactively and hence unattended, i.e. show no plots')

try:
    argcomplete.autocomplete(parser)
except NameError:
    pass


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


def pdg_to_name_faulty(pdg_code):
    """Return the particle name for a given Particle Data Group (PDG) code honoring a bug in the float to integer conversion.

    Args:
        pdg_code: PDG code compatible with the values in root files.

    Returns:
        The name of the particle for the given PDG code with 'None' as value for faulty reconstructions and 'nan' for buggy translations.

    """
    if pdg_code == 0:
        return 'None'

    try:
        return pdg.to_name(int(pdg_code))
    except LookupError:
        return 'nan'


def pyplot_sanitize_show(title, format='pdf', bbox_inches='tight', **kwargs):
    """Save (not /dev/null) and show (interactive) the current figure given an arbitrary title to a configurable location and sanitize its name.

    Args:
        title: Title of the plot and baseline for the name of the file.
        format: Format in which to save the plot; Its value is also appended to the filename.
        bbox_inches: Bbox in inches; If 'tight' figure out the best suitable values.

        Any keyword arguments valid for `matplotlib.pyplot.savefig()`.

    """
    output_directory = args.output_directory
    if output_directory != '/dev/null':
        if not os.path.exists(output_directory):
            print('Creating desired output directory "%s"'%(output_directory), file=sys.stderr)
            os.makedirs(output_directory, exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation

        title = re.sub('[\\\\$_^{}]', '', title)
        plt.savefig(output_directory + '/' + title + '.' + format, bbox_inches=bbox_inches, format=format, **kwargs)

    interactive = args.interactive
    if interactive:
        plt.show(block=False)
    else:
        plt.close()


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
            stat[p]['tpr'] = np.append(stat[p]['tpr'], [np.float64(particle_data[((particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] > cut)].shape[0]) / np.float64(particle_data[(particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0])])
            stat[p]['fpr'] = np.append(stat[p]['fpr'], [np.float64(particle_data[((particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] > cut)].shape[0]) / np.float64(particle_data[(particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))].shape[0])])
            stat[p]['tnr'] = np.append(stat[p]['tnr'], [np.float64(particle_data[((particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] <= cut)].shape[0]) / np.float64(particle_data[(particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))].shape[0])])
            stat[p]['ppv'] = np.append(stat[p]['ppv'], [np.float64(particle_data[((particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] > cut)].shape[0]) / np.float64(particle_data[particle_data[cutting_columns[p]] > cut].shape[0])])

            if not np.isclose(stat[p]['fpr'][-1]+stat[p]['tnr'][-1], 1, atol=1e-2):
                print('VALUES INCONSISTENT: ', end='')

            print('Particle %10s: TPR=%6.6f; FPR=%6.6f; TNR=%6.6f; PPV=%6.6f; cut=%4.4f'%(p, stat[p]['tpr'][-1], stat[p]['fpr'][-1], stat[p]['tnr'][-1], stat[p]['ppv'][-1], cut))

    return stat


def epsilonPID_matrix(cut=0.2, cutting_columns=particleIDs):
    """Calculate the epsilon_PID matrix for misclassifying particles: rows represent true particles, columns the classification.

    Args:
        cut: Position of the cut for the cutting_columns.
        cutting_columns: Dictionary which yields a column name for each particle on which the cuts are performed.

    Returns:
        A numpy matrix of epsilon_PID values. The `epsilon_PID[i][j]` value being the probability given it is a particle ''i'' that it will be categorized as particle ''j''.

    """
    epsilonPIDs = np.zeros(shape=(len(data.keys()), len(cutting_columns.keys())))
    for i, i_p in enumerate(data.keys()):
        for j, j_p in enumerate(cutting_columns.keys()):
            # The deuterium code is not properly stored in the mcPDG variable, hence the use of `pdg_from_name_faulty()`
            epsilonPIDs[i][j] = np.float64(data[i_p][((data[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) | (data[i_p]['mcPDG'] == -1 * pdg_from_name_faulty(i_p))) & (data[i_p][cutting_columns[j_p]] > cut)].shape[0]) / np.float64(data[i_p][(data[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) | (data[i_p]['mcPDG'] == -1 * pdg_from_name_faulty(i_p))].shape[0])

    print("epsilon_PID matrix:\n%s"%(epsilonPIDs))
    return epsilonPIDs


def mimic_pid(detector_weights=detector_weights, check=True):
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


def bayes(priors=defaultdict(lambda: 1., {}), detector='all', mc_best=False):
    """Compute probabilities for particle hypothesis using a Bayesian approach.

    Args:
        priors: Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.
        mc_best: Boolean specifying whether to use the Monte Carlo data for calculating the a prior probabilities.

    Returns:
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.

    """
    cutting_columns = {k: 'bayes_' + v for k, v in particleIDs.items()}

    for particle_data in data.values():
        # TODO: Use mimic_pid here to allow for weighted detector

        if mc_best == True:
            priors = {p: particle_data[(particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0] for p in particles}

        for p in particles:
            denominator = 0.
            for p_2 in particles:
                denominator += (particle_data['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - particle_data['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

            # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
            particle_data[cutting_columns[p]] = priors[p] / denominator

    return cutting_columns


def multivariate_bayes(holdings=['pt'], nbins=10, detector='all', mc_best=False, niterations=7, norm='pi+', whis=None):
    """Compute probabilities for particle hypothesis keeping the `hold` root variable fixed using a Bayesian approach.

    Args:
        holdings: List of Root variables on which the 'a prior' probability shall depend on.
        nbins: Number of bins to use for the `hold` variable when calculating probabilities.
        detector: Name of the detector to be used for pidLogLikelihood extraction.
        mc_best: Boolean specifying whether to use the Monte Carlo data for prior probabilities or an iterative approach.
        niterations: Number of iterations for the converging approach.
        norm: Particle by which abundance to norm the a priori probabilities.
        whis: Whiskers, scale of the Inter Quartile Range (IQR) for outlier exclusion.

    Returns:
        cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.
        cutting_columns_isMax: A dictionary containing the name of each column by particle which shall be used for exclusive cuts, yielding the maximum probability particle.
        category_columns: A dictionary of entries for each `hold` with the name of the column in each dataframe which holds the category for bin selection.
        intervals: A dictionary of entries for each `hold` containing an array of interval boundaries for every bin.
        iteration_priors: A dictionary for each dataset of particle dictionaries containing arrays of priors for each iteration.

    """
    if mc_best == True:
        niterations = 1

    category_columns = {hold: 'category_' + hold for hold in holdings}
    intervals = {hold: {} for hold in holdings}
    for p, particle_data in data.items():
        selection = np.ones(particle_data.shape[0], dtype=bool)
        if whis:
            for hold in holdings:
                q75, q25 = np.percentile(particle_data[hold], [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - (iqr * whis)
                upper_bound = q75 + (iqr * whis)
                selection = selection & (particle_data[hold] > lower_bound) & (particle_data[hold] < upper_bound)

        for hold in holdings:
            particle_data[category_columns[hold]], intervals[hold][p] = pd.qcut(particle_data[selection][hold], q=nbins, labels=range(nbins), retbins=True)

    cutting_columns = {k: 'bayes_' + '_'.join([str(hold) for hold in np.unique(holdings)]) + '_' + v for k, v in particleIDs.items()}
    iteration_priors = {l: {p: [[] for _ in range(niterations)] for p in particles} for l in particles}

    for l, particle_data in data.items():
        for i in itertools.product(*[range(nbins) for _ in range(len(holdings))]):
            selection = np.ones(particle_data.shape[0], dtype=bool)
            for m in range(len(holdings)):
                selection = selection & (particle_data[category_columns[holdings[m]]] == i[m])

            if mc_best == True:
                y = {p: np.float64(particle_data[selection & ((particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p)))].shape[0]) for p in particles}
                priors = {p: y[p] / y[norm] for p in particles}

                print('Priors ', holdings, ' at ', i, ' of ' + str(nbins) + ': ', priors)
            else:
                priors = {p: 1. for p in particles}

            for iteration in range(niterations):
                # Calculate the 'a posteriori' probability for each pt bin
                for p in particles:
                    denominator = 0.
                    for p_2 in particles:
                        denominator += (particle_data[selection]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - particle_data[selection]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

                    # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
                    particle_data.at[selection, cutting_columns[p]] = priors[p] / denominator

                y = {p: np.float64(particle_data[selection][cutting_columns[p]].sum()) for p in particles}
                for p in particles:
                    priors[p] = y[p] / y[norm]
                    iteration_priors[l][p][iteration] += [priors[p]]

                if not mc_best: print('Priors ', holdings, ' at ', i, ' of ' + str(nbins) + ' after %2d: '%(iteration + 1), priors)

    return cutting_columns, category_columns, intervals, iteration_priors

def add_isMax_column(cutting_columns):
    """Add columns containing ones for each track where the cutting column is maximal and fill zeros otherwise.

    Args:
        cutting_columns: Columns by particles where to find the maximum

    Returns:
        cutting_columns_isMax: Columns by particle containing ones for maximal values

    """
    for particle_data in data.values():
        max_columns = particle_data[list(cutting_columns.values())].idxmax(axis=1)
        cutting_columns_isMax = {k: v + '_isMax' for k, v in cutting_columns.items()}

        for p in cutting_columns.keys():
            particle_data[cutting_columns_isMax[p]] = np.where(max_columns == cutting_columns[p], 1, 0)

    return cutting_columns_isMax


def plot_stats_by_particle(stat, particles_of_interest=particles):
    for p in particles_of_interest:
        plt.figure()
        plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='ROC')
        # Due to the fact that FPR + TNR = 1 the plot will simply show a straight line; Use for debugging only
        # plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
        plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='PPV')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Particle Rates')
        drawing_title = plt.title('%s Identification'%(particle_base_formats[p]))
        plt.legend()
        pyplot_sanitize_show('Statistics: ' + drawing_title.get_text())


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
        plt.xticks(range(len(particles)), [particle_base_formats[p] for p in particles])
        plt.ylabel('True Particle')
        plt.yticks(range(len(particles)), [particle_base_formats[p] for p in particles])
        plt.title('ID' + title_suffixes[n])
        plt.tight_layout(pad=1.4)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.20, 0.05, 0.6])
    plt.colorbar(cax=cbar_ax)
    pyplot_sanitize_show('Diff Heatmap: ' + drawing_title.get_text() + ','.join(str(suffix) for suffix in title_suffixes))


def plot_diff_stats(stats_approaches=[], title_suffixes=[], particles_of_interest=particles):
    if len(stats_approaches) >= 0 and len(stats_approaches) != len(title_suffixes):
        raise ValueError('stats_approaches array must be of same length as the title_suffixes array')

    for p in particles_of_interest:
        plt.figure()
        grid = plt.GridSpec(3, 1, hspace=0.1)

        main_ax = plt.subplot(grid[:2, 0])
        drawing_title = plt.title('%s Identification'%(particle_base_formats[p]))
        for n, approach in enumerate(stats_approaches):
            drawing = plt.plot(approach[p]['fpr'], approach[p]['tpr'], label='ROC' + title_suffixes[n])
            plt.plot(approach[p]['fpr'], approach[p]['ppv'], label='PPV' + title_suffixes[n], linestyle=':', color=drawing[0].get_color())

        plt.setp(main_ax.get_xticklabels(), visible=False)
        plt.ylabel('Particle Rates')
        plt.legend()

        plt.subplot(grid[2, 0], sharex=main_ax)
        base_approach = stats_approaches[0]
        for n, approach in enumerate(stats_approaches[1:], 1):
            sorted_range = np.argsort(approach[p]['fpr']) # Numpy expects values sorted by x
            interpolated_rate = np.interp(base_approach[p]['fpr'], approach[p]['fpr'][sorted_range], approach[p]['tpr'][sorted_range])
            ratio = np.divide(interpolated_rate, base_approach[p]['tpr'], out=np.ones_like(interpolated_rate), where=base_approach[p]['tpr']!=0)
            plt.plot(base_approach[p]['fpr'], ratio, label='TPR%s /%s'%(title_suffixes[n], title_suffixes[0]), color='C2')

            sorted_range = np.argsort(approach[p]['fpr']) # Numpy expects values sorted by x
            interpolated_rate = np.interp(base_approach[p]['fpr'], approach[p]['fpr'][sorted_range], approach[p]['ppv'][sorted_range])
            ratio = np.divide(interpolated_rate, base_approach[p]['ppv'], out=np.ones_like(interpolated_rate), where=base_approach[p]['ppv']!=0)
            plt.plot(base_approach[p]['fpr'], ratio, label='PPV%s /%s'%(title_suffixes[n], title_suffixes[0]), linestyle=':', color='C3')

        plt.axhline(y=1., color='dimgrey', linestyle='--')
        plt.grid(b=True, axis='both')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Rate Ratios')
        plt.legend()

        pyplot_sanitize_show('Diff Statistics: ' + drawing_title.get_text() + ','.join(str(suffix) for suffix in title_suffixes))


args = parser.parse_args()

# Read in all the particle's information into a dictionary of pandas-frames
input_directory =  args.input_directory
data = {p: rpd.read_root(input_directory + '/' + p + '.root') for p in particles}
# Clean up the data; Remove obviously un-physical values
for particle_data in data.values():
    for k, bounds in physical_boundaries.items():
        particle_data.drop(particle_data[(particle_data[k] < bounds[0]) | (particle_data[k] > bounds[1])].index, inplace=True)

if args.run_stats:
    nbins = args.nbins
    particles_of_interest = args.particles_of_interest

    detector = 'all'

    for p in particles_of_interest:
        # Abundances might vary due to some preliminary mass hypothesis being applied on reconstruction, hence plot for each dataset
        particle_data = data[p]

        unique_particles = np.unique(particle_data['mcPDG'].values)
        true_abundance = np.array([particle_data[particle_data['mcPDG'] == code].shape[0] for code in unique_particles])
        sorted_range = np.argsort(true_abundance)
        true_abundance = true_abundance[sorted_range][::-1]
        unique_particles = unique_particles[sorted_range][::-1]

        plt.figure()
        plt.grid(b=False, axis='x')
        plt.errorbar(range(len(unique_particles)), true_abundance, xerr=0.5, fmt='o')
        plt.xticks(range(len(unique_particles)), [particle_formats[pdg_to_name_faulty(k)] for k in unique_particles])
        drawing_title = plt.title('True Particle Abundances in the reconstructed Decays')
        pyplot_sanitize_show('General Purpose Statistics: ' + drawing_title.get_text())

        likelihood_ratio_bins, intervals = pd.cut(particle_data['pidProbabilityExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc'], nbins, labels=range(nbins), retbins=True)
        abundance_ratio = np.zeros(nbins)
        y_err = np.zeros(nbins)
        for i in range(nbins):
            particle_data_bin = particle_data[likelihood_ratio_bins == i]
            numerator = particle_data_bin[(particle_data_bin['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0]
            denominator = np.array([particle_data_bin[(particle_data_bin['mcPDG'] == pdg_from_name_faulty(p_2)) | (particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p_2))].shape[0] for p_2 in particles]).sum()

            abundance_ratio[i] = numerator / denominator
            # Due to the correlation of of abundance_ratio and likelihood_ratio_bins the actual error is (?) y_err = sqrt(abundance_ratio) * efficiency * (1 - efficiency)
            # For now use Gaussian Error Propagation, despite not really being appropriate (TODO)
            y_err[i] = abundance_ratio[i] * np.sqrt(np.power((np.sqrt(numerator)/numerator), 2) + np.power((np.sqrt(denominator)/denominator), 2))

        interval_centers = np.array([np.mean(intervals[i:i+2]) for i in range(len(intervals)-1)])
        # As xerr np.array([intervals[i] - intervals[i-1] for i in range(1, len(intervals))]) / 2. may be used, indicating the width of the bins

        plt.figure()
        plt.errorbar(interval_centers, abundance_ratio, yerr=y_err, capsize=3, elinewidth=1, marker='o', markersize=4, markeredgewidth=1, markerfacecolor='None', linestyle='--', linewidth=0.2)
        drawing_title = plt.title('Relative Abundance in Likelihood Ratio Bins')
        plt.xlabel('%s Likelihood Ratio'%(particle_base_formats[p]))
        plt.ylabel('Relative %s Abundance'%(particle_base_formats[p]))
        plt.ylim(-0.05, 1.05)
        pyplot_sanitize_show('General Purpose Statistics: ' + drawing_title.get_text())

if args.run_pid:
    cut = args.cut
    exclusive_cut = args.exclusive_cut

    particles_of_interest = args.particles_of_interest

    plot_stats_by_particle(stats(), particles_of_interest=particles_of_interest)

    c = add_isMax_column(particleIDs) if exclusive_cut else particleIDs
    epsilonPIDs = epsilonPID_matrix(cutting_columns=c, cut=cut)
    plt.figure()
    plt.imshow(epsilonPIDs, cmap='viridis')
    for (j, i), label in np.ndenumerate(epsilonPIDs):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
    plt.grid(b=False, axis='both')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_base_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_base_formats[p] for p in particles])
    plt.colorbar()
    if exclusive_cut:
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut')
    else:
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut))
    pyplot_sanitize_show('Particle ID Approach: ' + drawing_title.get_text())

if args.run_mimic_pid:
    mimic_pid()

if args.run_bayes:
    mc_best = args.mc_best
    particles_of_interest = args.particles_of_interest

    c = bayes(mc_best=mc_best)
    plot_stats_by_particle(stats(cutting_columns=c), particles_of_interest=particles_of_interest)

if args.diff_methods:
    methods = args.diff_methods
    assert (len(methods) >= 2), 'Specify at least two methods'

    cut = args.cut
    ncuts = args.ncuts

    hold = args.hold
    holdings = args.holdings
    whis = args.whis
    nbins = args.nbins
    niterations = args.niterations
    norm = args.norm
    mc_best = args.mc_best
    exclusive_cut = args.exclusive_cut

    particles_of_interest = args.particles_of_interest

    epsilonPIDs_approaches = []
    stats_approaches = []
    title_suffixes = []
    for m in methods:
        if m == 'pid':
            c = particleIDs
            title_suffixes += [' via PID']
        elif m == 'flat_bayes':
            c = bayes()
            title_suffixes += [' via flat Bayes']
        elif m == 'simple_bayes':
            c = bayes(mc_best=True)
            title_suffixes += [' via simple Bayes']
        elif m == 'univariate_bayes':
            c = multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            title_suffixes += [' via univariate Bayes']
        elif m in ['univariate_bayes_by_' + v for v in variable_formats.keys()]:
            explicit_hold = re.sub('univariate_bayes_by_', '', m)
            c = multivariate_bayes(holdings=[explicit_hold], whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            title_suffixes += [' by ' + variable_formats[explicit_hold]]
        elif m == 'multivariate_bayes':
            c = multivariate_bayes(holdings=holdings, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            title_suffixes += [' by ' + ' & '.join([variable_formats[h] for h in holdings])]
        else:
            raise ValueError('received unknown method "%s"'%(m))

        c_choice = add_isMax_column(c) if exclusive_cut else c
        epsilonPIDs_approaches += [epsilonPID_matrix(cutting_columns=c_choice, cut=cut)]
        stats_approaches += [stats(cutting_columns=c, ncuts=ncuts)]

    if exclusive_cut:
        title_epsilonPIDs = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        title_epsilonPIDs = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    plot_diff_epsilonPIDs(epsilonPIDs_approaches=epsilonPIDs_approaches, title_suffixes=title_suffixes, title_epsilonPIDs=title_epsilonPIDs)
    plot_diff_stats(stats_approaches=stats_approaches, title_suffixes=title_suffixes, particles_of_interest=particles_of_interest)

if args.run_univariate_bayes:
    cut = args.cut

    hold = args.hold
    whis = args.whis
    niterations = args.niterations
    nbins = args.nbins
    norm = args.norm
    mc_best = args.mc_best
    exclusive_cut = args.exclusive_cut
    cutting_columns, category_columns, intervals, _ = multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals[hold].items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals[hold].items()}

    particles_of_interest = args.particles_of_interest

    plt.figure()
    drawing_title = plt.title('True Positive Rate for a Cut at %.2f'%(cut))
    for p in particles_of_interest:
        assumed_abundance = np.array([data[p][((data[p]['mcPDG'] == pdg_from_name_faulty(p)) | (data[p]['mcPDG'] == -1 * pdg_from_name_faulty(p))) & (data[p][category_columns[hold]] == it) & (data[p][cutting_columns[p]] > cut)].shape[0] for it in range(nbins)])
        actual_abundance = np.array([data[p][((data[p]['mcPDG'] == pdg_from_name_faulty(p)) | (data[p]['mcPDG'] == -1 * pdg_from_name_faulty(p))) & (data[p][category_columns[hold]] == it)].shape[0] for it in range(nbins)])
        plt.errorbar(interval_centers[p], assumed_abundance / actual_abundance, xerr=interval_widths[p], label='%s'%(particle_base_formats[p]), fmt='o')

    plt.xlabel(variable_formats[hold] + ' (' + variable_units[hold] + ')')
    plt.ylabel('True Positive Rate')
    plt.legend()
    pyplot_sanitize_show('Univariate Bayesian Approach: ' + drawing_title.get_text())

    c = add_isMax_column(cutting_columns) if exclusive_cut else cutting_columns
    epsilonPIDs = epsilonPID_matrix(cutting_columns=c, cut=cut)
    plt.figure()
    plt.imshow(epsilonPIDs, cmap='viridis')
    for (j, i), label in np.ndenumerate(epsilonPIDs):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
    plt.grid(b=False, axis='both')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_base_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_base_formats[p] for p in particles])
    if exclusive_cut:
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut')
    else:
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut))
    pyplot_sanitize_show('Univariate Bayesian Approach: ' + drawing_title.get_text())

    plot_stats_by_particle(stats(cutting_columns=cutting_columns), particles_of_interest=particles_of_interest)

if args.run_univariate_bayes_priors:
    particles_of_interest = args.particles_of_interest
    hold = args.hold
    whis = args.whis
    nbins = args.nbins
    niterations = args.niterations
    norm = args.norm

    iteration_priors_viaIter = multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)[-1]
    intervals, iteration_priors_viaBest = multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=True, nbins=nbins)[-2:]
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals[hold].items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals[hold].items()}

    for p in particles_of_interest:
        plt.figure()
        drawing_title = plt.title('%s Spectra Ratios Relative to %s'%(particle_base_formats[p], particle_base_formats[norm]))
        plt.errorbar(interval_centers[p], iteration_priors_viaBest[norm][p][-1], xerr=interval_widths[p], label='Truth', fmt='*')
        for n in range(niterations):
            plt.errorbar(interval_centers[p], iteration_priors_viaIter[norm][p][n], xerr=interval_widths[p], label='Iteration %d'%(n+1), fmt='o')

        plt.xlabel(variable_formats[hold] + ' (' + variable_units[hold] + ')')
        plt.ylabel('Relative Abundance')
        plt.legend()
        pyplot_sanitize_show('Univariate Bayesian Approach: ' + drawing_title.get_text())

if args.run_univariate_bayes_outliers:
    hold = args.hold
    whis = args.whis
    norm = args.norm

    plt.figure()
    if whis:
        plt.boxplot(data[norm][hold], whis=whis, sym='+')
    else:
        plt.boxplot(data[norm][hold], whis='range', sym='+')
    drawing_title = plt.title('Outliers Outside of ' + str(whis) + ' IQR on a Logarithmic Scale')
    plt.yscale('log')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel(variable_formats[hold] + ' (' + variable_units[hold] + ')')
    pyplot_sanitize_show('Univariate Bayesian Approach: ' + drawing_title.get_text())

if args.run_multivariate_bayes:
    cut = args.cut

    holdings = args.holdings
    mc_best = args.mc_best
    exclusive_cut = args.exclusive_cut
    whis = args.whis
    niterations = args.niterations
    nbins = args.nbins
    norm = args.norm
    particles_of_interest = args.particles_of_interest

    cutting_columns, category_columns, intervals, iteration_priors = multivariate_bayes(holdings=holdings, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)

    interval_centers = {}
    interval_widths = {}
    for hold in holdings:
        interval_centers[hold] = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals[hold].items()}
        interval_widths[hold] = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals[hold].items()}

    for p in particles_of_interest:
        fig = plt.figure()
        plt.imshow(np.array(iteration_priors[norm][p]).reshape(nbins, nbins).T, cmap='viridis')
        plt.grid(b=False, axis='both')
        plt.xlabel(variable_formats[holdings[0]] + ' (' + variable_units[holdings[0]] + ')')
        plt.xticks(range(nbins), interval_centers[holdings[0]][p])
        fig.autofmt_xdate()
        plt.ylabel(variable_formats[holdings[1]] + ' (' + variable_units[holdings[1]] + ')')
        plt.yticks(range(nbins), interval_centers[holdings[1]][p])
        drawing_title = plt.title('%s Spectra Ratios Relative to %s'%(particle_base_formats[p], particle_base_formats[norm]))
        plt.colorbar()
        pyplot_sanitize_show('Multivariate Bayesian Approach: ' + drawing_title.get_text())

    c = add_isMax_column(cutting_columns) if exclusive_cut else cutting_columns
    epsilonPIDs = epsilonPID_matrix(cutting_columns=c, cut=cut)
    plt.figure()
    plt.imshow(epsilonPIDs, cmap='viridis')
    for (j, i), label in np.ndenumerate(epsilonPIDs):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
    plt.grid(b=False, axis='both')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(particles)), [particle_base_formats[p] for p in particles])
    plt.ylabel('True Particle')
    plt.yticks(range(len(particles)), [particle_base_formats[p] for p in particles])
    if exclusive_cut:
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut')
    else:
        drawing_title = plt.title(r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut))
    plt.colorbar()
    pyplot_sanitize_show('Multivariate Bayesian Approach: ' + drawing_title.get_text())

    plot_stats_by_particle(stats(cutting_columns=cutting_columns), particles_of_interest=particles_of_interest)

if args.run_multivariate_bayes_motivation:
    norm = args.norm
    nbins = args.nbins
    whis = args.whis
    particles_of_interest = args.particles_of_interest

    holdings = args.holdings
    particle_data = data[norm]

    truth_color_column = 'mcPDG_color'

    correlation_matrix = particle_data[holdings].corr()
    plt.figure()
    plt.imshow(correlation_matrix, cmap='viridis')
    for (j, i), label in np.ndenumerate(correlation_matrix):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small')
    plt.grid(b=False, axis='both')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(holdings)), [variable_formats[v] for v in holdings])
    plt.ylabel('True Particle')
    plt.yticks(range(len(holdings)), [variable_formats[v] for v in holdings])
    plt.colorbar()
    drawing_title = plt.title('Heatmap of Correlation Matrix of Root Variables')
    pyplot_sanitize_show('Multivariate Bayesian Approach: ' + drawing_title.get_text())

    selection = np.ones(particle_data.shape[0], dtype=bool)
    if whis:
        for hold in holdings:
            q75, q25 = np.percentile(particle_data[hold], [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (iqr * whis)
            upper_bound = q75 + (iqr * whis)
            selection = selection & (particle_data[hold] > lower_bound) & (particle_data[hold] < upper_bound)

    fig = plt.figure(figsize=(6, 6))
    drawing_title = plt.suptitle('Multi-axes Histogram of ' + ', '.join(format(hold) for hold in holdings))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    main_ax = plt.subplot(grid[:-1, 1:])
    for v in np.unique(particle_data[selection]['mcPDG'].values):
        particle_data.at[selection & (particle_data[selection]['mcPDG'] == v), truth_color_column] = list(np.unique(particle_data[selection]['mcPDG'].values)).index(v)
    plt.scatter(particle_data[selection][holdings[0]], particle_data[selection][holdings[1]], c=particle_data[selection][truth_color_column], cmap=plt.cm.get_cmap('viridis', np.unique(particle_data[selection]['mcPDG'].values).shape[0]), s=5., alpha=.1)
    plt.setp(main_ax.get_xticklabels(), visible=False)
    plt.setp(main_ax.get_yticklabels(), visible=False)

    plt.subplot(grid[-1, 1:], sharex=main_ax)
    plt.hist(particle_data[selection][holdings[0]], nbins, histtype='step', orientation='vertical')
    plt.gca().invert_yaxis()
    plt.xlabel(variable_formats[holdings[0]] + ' (' + variable_units[holdings[0]] + ')')
    plt.subplot(grid[:-1, 0], sharey=main_ax)
    plt.hist(particle_data[selection][holdings[1]], nbins, histtype='step', orientation='horizontal')
    plt.gca().invert_xaxis()
    plt.ylabel(variable_formats[holdings[1]] + ' (' + variable_units[holdings[1]] + ')')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.20, 0.05, 0.6])
    cbar = plt.colorbar(cax=cbar_ax, ticks=range(np.unique(particle_data[selection]['mcPDG'].values).shape[0]))
    cbar.set_alpha(1.)
    cbar.set_ticklabels([particle_formats[pdg_to_name_faulty(v)] for v in np.unique(particle_data[selection]['mcPDG'].values)])
    cbar.draw_all()

    pyplot_sanitize_show('Multivariate Bayesian Approach: ' + drawing_title.get_text())


plt.show()
