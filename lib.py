#!/usr/bin/env python3

import argparse
import itertools
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import root_pandas as rpd

import pdg


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


class ParticleFrame(dict):
    """ParticleFrame in analogy to pandas' DataFrame but for Particles.

    Mimic the behavior of a dictionary containing the particle data as values but include some utility functions and base definitions of stable particles and detector components.

    Attributes:
        particles (list): List of stable particles.
        particleIDs (dict): Particle IDs according to the generic ID process for stable particles.
        particle_formats (dict): Format string in a dictionary of stable and unstable particles and anti-particles.
        particle_base_formats (dict): Format string of the base particle for each stable and unstable particles and anti-particles.
        detectors (list): List of real detectors.
        pseudo_detectors (list): List of pseudo detectors which are composed of real detectors.
        variable_formats (dict): Format strings for ROOT variables.
        variable_units (dict): Format strings for the unit of ROOT variables.
        detector_weights (dict): Relative weights of detectors.
        physical_boundaries (dict): Boundaries for root variables according to which the ParticleFrame is cut to exclude un-physical results.

    """
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

    def __init__(self, input_directory=None, output_directory=None, interactive=None):
        """Initialize and empty ParticleFrame.

        Args:
            input_directory (:obj:`str`, optional): Default input directory for root files for each particle.
            output_directory (:obj:`str`, optional): Default output directory for data generated using this ParticleFrame.
            interactive (:obj:`bool`, optional): Whether plotting should be done interactively.

        """
        self.data = {}
        if input_directory is not None:
            self.read_root(input_directory)
        self.output_directory = './res/' if output_directory is None else output_directory
        self.interactive = False if interactive is None else interactive

    def __setitem__(self, key, item):
        self.data[key] = item

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def __delitem__(self, key):
        del self.data[key]

    def clear(self):
        return self.data.clear()

    def copy(self):
        return self.data.copy()

    def has_key(self, k):
        return k in self.data

    def update(self, *args, **kwargs):
        return self.data.update(*args, **kwargs)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def pop(self, *args):
        return self.data.pop(*args)

    def __cmp__(self, dict_):
        return self.data.__cmp__(self.data, dict_)

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        return iter(self.data)

    def read_root(self, input_directory):
        """Read in the particle information contained within the given directory into the current object.

        Args:
            input_directory (str): Directory in which the program shall search for root files for each particle

        """
        # Read in all the particle's information into a dictionary of pandas-frames
        self.data = {p: rpd.read_root(input_directory + '/' + p + '.root') for p in self.particles}
        # Clean up the data; Remove obviously un-physical values
        for particle_data in self.data.values():
            for k, bounds in self.physical_boundaries.items():
                particle_data.drop(particle_data[(particle_data[k] < bounds[0]) | (
                    particle_data[k] > bounds[1])].index, inplace=True)

    def stats(self, cut_min=0., cut_max=1., ncuts=50, cutting_columns=None):
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
        cutting_columns = self.particleIDs if cutting_columns is None else cutting_columns

        stat = {}
        cuts = np.linspace(cut_min, cut_max, num=ncuts)
        for p, particle_data in self.items():
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

    def epsilonPID_matrix(self, cut=0.2, cutting_columns=None):
        """Calculate the epsilon_PID matrix for misclassifying particles: rows represent true particles, columns the classification.

        Args:
            cut: Position of the cut for the cutting_columns.
            cutting_columns: Dictionary which yields a column name for each particle on which the cuts are performed.

        Returns:
            A numpy matrix of epsilon_PID values. The `epsilon_PID[i][j]` value being the probability given it is a particle ''i'' that it will be categorized as particle ''j''.

        """
        cutting_columns = self.particleIDs if cutting_columns is None else cutting_columns

        epsilonPIDs = np.zeros(shape=(len(self.keys()), len(cutting_columns.keys())))
        for i, i_p in enumerate(self.keys()):
            for j, j_p in enumerate(cutting_columns.keys()):
                # The deuterium code is not properly stored in the mcPDG variable, hence the use of `pdg_from_name_faulty()`
                epsilonPIDs[i][j] = np.float64(self[i_p][((self[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) | (self[i_p]['mcPDG'] == -1 * pdg_from_name_faulty(i_p))) & (self[i_p][cutting_columns[j_p]] > cut)].shape[0]) / np.float64(self[i_p][(self[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) | (self[i_p]['mcPDG'] == -1 * pdg_from_name_faulty(i_p))].shape[0])

        print("epsilon_PID matrix:\n%s"%(epsilonPIDs))
        return epsilonPIDs

    def mimic_pid(self, detector_weights=None, check=True):
        """Mimic the calculation of the particleIDs and compare them to their value provided by the analysis software.

        Args:
            detector_weights: Dictionary of detectors with the weights (default 1.) as values.
            check: Whether to assert the particleIDs if the detector weights are all 1.

        """
        detector_weights = self.detector_weights if detector_weights is None else detector_weights

        if not all(v == 1. for v in detector_weights.values()):
            check = False

        for particle_data in self.values():
            # Calculate the accumulated logLikelihood and assess the relation to kaonID
            for p in self.particles:
                particle_data['accumulatedLogLikelihood' + basf2_Code(p)] = np.zeros(particle_data[self.particleIDs[p]].shape[0])
                # The following loop is equivalent to querying the 'all' pseudo-detector when using flat detector weights
                for d in self.detectors:
                    column = 'pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + d + '__bc'
                    # Fill up NaN values for detectors which do not yield a result
                    # Since at least one detector must return a logLikelihood it is not possible that only NaN values lead to a probability of 1
                    particle_data['accumulatedLogLikelihood' + basf2_Code(p)] += particle_data[column].fillna(0) * detector_weights[d]

            # Calculate the particleIDs manually and compare them to the result of the analysis software
            particle_data['assumed_' + self.particleIDs['pi+']] = 1. / (1. + (particle_data['accumulatedLogLikelihood' + basf2_Code('K+')] - particle_data['accumulatedLogLikelihood' + basf2_Code('pi+')]).apply(np.exp))
            for p in (set(self.particles) - set(['pi+'])):
                # Algebraic trick to make exp(a)/(exp(a) + exp(b)) stable even for very small values of a and b
                particle_data['assumed_' + self.particleIDs[p]] = 1. / (1. + (particle_data['accumulatedLogLikelihood' + basf2_Code('pi+')] - particle_data['accumulatedLogLikelihood' + basf2_Code(p)]).apply(np.exp))

                if check:
                    # Assert for equality of the manual calculation and analysis software's output
                    npt.assert_allclose(particle_data['assumed_' + self.particleIDs[p]].values, particle_data[self.particleIDs[p]].astype(np.float64).values, atol=1e-3)

        print('Successfully calculated the particleIDs using the logLikelihoods only.')

    def bayes(self, priors=defaultdict(lambda: 1., {}), detector='all', mc_best=False):
        """Compute probabilities for particle hypothesis using a Bayesian approach.

        Args:
            priors: Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.
            mc_best: Boolean specifying whether to use the Monte Carlo data for calculating the a prior probabilities.

        Returns:
            cutting_columns: A dictionary containing the name of each column by particle which shall be used for cuts.

        """
        cutting_columns = {k: 'bayes_' + v for k, v in self.particleIDs.items()}

        for particle_data in self.values():
            # TODO: Use mimic_pid here to allow for weighted detector

            if mc_best == True:
                priors = {p: particle_data[(particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0] for p in self.particles}

            for p in self.particles:
                denominator = 0.
                for p_2 in self.particles:
                    denominator += (particle_data['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - particle_data['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

                # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
                particle_data[cutting_columns[p]] = priors[p] / denominator

        return cutting_columns

    def multivariate_bayes(self, holdings=['pt'], nbins=10, detector='all', mc_best=False, niterations=7, norm='pi+', whis=None):
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
        for p, particle_data in self.items():
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

        cutting_columns = {k: 'bayes_' + '_'.join([str(hold) for hold in np.unique(holdings)]) + '_' + v for k, v in self.particleIDs.items()}
        iteration_priors = {l: {p: [[] for _ in range(niterations)] for p in self.particles} for l in self.particles}

        for l, particle_data in self.items():
            for i in itertools.product(*[range(nbins) for _ in range(len(holdings))]):
                selection = np.ones(particle_data.shape[0], dtype=bool)
                for m in range(len(holdings)):
                    selection = selection & (particle_data[category_columns[holdings[m]]] == i[m])

                if mc_best == True:
                    y = {p: np.float64(particle_data[selection & ((particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p)))].shape[0]) for p in self.particles}
                    priors = {p: y[p] / y[norm] for p in self.particles}

                    print('Priors ', holdings, ' at ', i, ' of ' + str(nbins) + ': ', priors)
                else:
                    priors = {p: 1. for p in self.particles}

                for iteration in range(niterations):
                    # Calculate the 'a posteriori' probability for each pt bin
                    for p in self.particles:
                        denominator = 0.
                        for p_2 in self.particles:
                            denominator += (particle_data[selection]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p_2) + '__cm__sp' + detector + '__bc'] - particle_data[selection]['pidLogLikelihoodValueExpert__bo' + basf2_Code(p) + '__cm__sp' + detector + '__bc']).apply(np.exp) * priors[p_2]

                        # Algebraic trick to make exp(H_i)*C_i/sum(exp(H_k) * C_k, k) stable even for very small values of H_i and H_k
                        particle_data.at[selection, cutting_columns[p]] = priors[p] / denominator

                    y = {p: np.float64(particle_data[selection][cutting_columns[p]].sum()) for p in self.particles}
                    for p in self.particles:
                        priors[p] = y[p] / y[norm]
                        iteration_priors[l][p][iteration] += [priors[p]]

                    if not mc_best: print('Priors ', holdings, ' at ', i, ' of ' + str(nbins) + ' after %2d: '%(iteration + 1), priors)

        return cutting_columns, category_columns, intervals, iteration_priors

    def add_isMax_column(self, cutting_columns):
        """Add columns containing ones for each track where the cutting column is maximal and fill zeros otherwise.

        Args:
            cutting_columns: Columns by particles where to find the maximum

        Returns:
            cutting_columns_isMax: Columns by particle containing ones for maximal values

        """
        for particle_data in self.values():
            max_columns = particle_data[list(cutting_columns.values())].idxmax(axis=1)
            cutting_columns_isMax = {k: v + '_isMax' for k, v in cutting_columns.items()}

            for p in cutting_columns.keys():
                particle_data[cutting_columns_isMax[p]] = np.where(max_columns == cutting_columns[p], 1, 0)

        return cutting_columns_isMax

    def pyplot_sanitize_show(self, title, format='pdf', bbox_inches='tight', output_directory=None, interactive=None, **kwargs):
        """Save and show the current figure to a configurable location and sanitize its name.

        Args:
            title (str): Title of the plot and baseline for the name of the file.
            format (:obj:`str`, optional): Format in which to save the plot; Its value is also appended to the filename.
            bbox_inches (:obj:`str`, optional): Bbox in inches; If 'tight' figure out the best suitable values.
            output_directory (:obj:`str`, optional): Output directory; Defaulting to the ParticleFrame's `self.output_directory`, do not save anything if given '/dev/null' as output.
            interactive (:obj:`str`, optional): Whether to run interactive or not; Defaulting to the ParticleFrame's `self.interactive`.
            **kwargs: Any keyword arguments valid for `matplotlib.pyplot.savefig()`.

        """
        output_directory = self.output_directory if output_directory is None else output_directory
        interactive = self.interactive if interactive is None else interactive

        if output_directory != '/dev/null':
            if not os.path.exists(output_directory):
                print('Creating desired output directory "%s"'%(output_directory), file=sys.stderr)
                os.makedirs(output_directory, exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation

            title = re.sub('[\\\\$_^{}]', '', title)
            plt.savefig(output_directory + '/' + title + '.' + format, bbox_inches=bbox_inches, format=format, **kwargs)

        if interactive:
            plt.show(block=False)
        else:
            plt.close()

    def plot_stats_by_particle(self, stat, particles_of_interest=None):
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        for p in particles_of_interest:
            plt.figure()
            plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='ROC')
            # Due to the fact that FPR + TNR = 1 the plot will simply show a straight line; Use for debugging only
            # plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
            plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='PPV')
            plt.xlabel('False Positive Rate')
            plt.ylabel('Particle Rates')
            drawing_title = plt.title('%s Identification'%(self.particle_base_formats[p]))
            plt.legend()
            self.pyplot_sanitize_show('Statistics: ' + drawing_title.get_text())

    def plot_neyman_pearson(self, nbins=10, cutting_columns=None, title_suffix='', particles_of_interest=None):
        cutting_columns = self.particleIDs if cutting_columns is None else cutting_columns
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        for p in particles_of_interest:
            particle_data = self[p]

            likelihood_ratio_bins, intervals = pd.cut(particle_data[cutting_columns[p]], nbins, labels=range(nbins), retbins=True)
            abundance_ratio = np.zeros(nbins)
            y_err = np.zeros(nbins)
            for i in range(nbins):
                particle_data_bin = particle_data[likelihood_ratio_bins == i]
                numerator = particle_data_bin[(particle_data_bin['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0]
                denominator = np.array([particle_data_bin[(particle_data_bin['mcPDG'] == pdg_from_name_faulty(p_2)) | (particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p_2))].shape[0] for p_2 in self.particles]).sum()

                abundance_ratio[i] = numerator / denominator
                y_err[i] = np.sqrt(abundance_ratio[i] * (1 - abundance_ratio[i]) / denominator)

            interval_centers = np.array([np.mean(intervals[i:i+2]) for i in range(len(intervals)-1)])
            # As xerr np.array([intervals[i] - intervals[i-1] for i in range(1, len(intervals))]) / 2. may be used, indicating the width of the bins

            plt.figure()
            plt.errorbar(interval_centers, abundance_ratio, yerr=y_err, capsize=3, elinewidth=1, marker='o', markersize=4, markeredgewidth=1, markerfacecolor='None', linestyle='--', linewidth=0.2)
            drawing_title = plt.title('Relative %s Abundance in Likelihood Ratio Bins%s'%(self.particle_base_formats[p], title_suffix))
            plt.xlabel('%s Likelihood Ratio'%(self.particle_base_formats[p]))
            plt.gcf().autofmt_xdate()
            plt.ylabel('Relative Abundance')
            plt.ylim(-0.05, 1.05)
            self.pyplot_sanitize_show('General Purpose Statistics: ' + drawing_title.get_text())

    def plot_diff_epsilonPIDs(self, epsilonPIDs_approaches=[], title_suffixes=[], title_epsilonPIDs=''):
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
            plt.xticks(range(len(self.particles)), [self.particle_base_formats[p] for p in self.particles])
            plt.ylabel('True Particle')
            plt.yticks(range(len(self.particles)), [self.particle_base_formats[p] for p in self.particles])
            plt.title('ID' + title_suffixes[n])
            plt.tight_layout(pad=1.4)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.20, 0.05, 0.6])
        plt.colorbar(cax=cbar_ax)
        self.pyplot_sanitize_show('Diff Heatmap: ' + drawing_title.get_text() + ','.join(str(suffix) for suffix in title_suffixes))

    def plot_diff_stats(self, stats_approaches=[], title_suffixes=[], particles_of_interest=None, ninterpolations=100):
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        if len(stats_approaches) >= 0 and len(stats_approaches) != len(title_suffixes):
            raise ValueError('stats_approaches array must be of same length as the title_suffixes array')

        for p in particles_of_interest:
            plt.figure()
            grid = plt.GridSpec(3, 1, hspace=0.1)
            colors = iter(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

            main_ax = plt.subplot(grid[:2, 0])
            drawing_title = plt.title('%s Identification'%(self.particle_base_formats[p]))
            for n, approach in enumerate(stats_approaches):
                drawing = plt.plot(approach[p]['fpr'], approach[p]['tpr'], label='ROC' + title_suffixes[n], color=next(colors))
                plt.plot(approach[p]['fpr'], approach[p]['ppv'], label='PPV' + title_suffixes[n], linestyle=':', color=drawing[0].get_color())

            plt.setp(main_ax.get_xticklabels(), visible=False)
            plt.ylabel('Particle Rates')
            plt.legend()

            plt.subplot(grid[2, 0], sharex=main_ax)
            base_approach = stats_approaches[0]
            for n, approach in enumerate(stats_approaches[1:], 1):
                x = np.linspace(np.sort(base_approach[p]['fpr'])[1], max(base_approach[p]['fpr']), ninterpolations) # Skip first value FPR value (zero)

                sorted_range = np.argsort(approach[p]['fpr']) # Numpy expects values sorted by x
                interpolated_rate = np.interp(x, approach[p]['fpr'][sorted_range], approach[p]['tpr'][sorted_range])
                sorted_range = np.argsort(base_approach[p]['fpr']) # Numpy expects values sorted by x
                interpolated_rate_base = np.interp(x, base_approach[p]['fpr'][sorted_range], base_approach[p]['tpr'][sorted_range])
                plt.plot(x, interpolated_rate / interpolated_rate_base, label='TPR%s /%s'%(title_suffixes[n], title_suffixes[0]), color=next(colors))

                sorted_range = np.argsort(approach[p]['fpr']) # Numpy expects values sorted by x
                interpolated_rate = np.interp(x, approach[p]['fpr'][sorted_range], approach[p]['ppv'][sorted_range])
                sorted_range = np.argsort(base_approach[p]['fpr']) # Numpy expects values sorted by x
                interpolated_rate_base = np.interp(x, base_approach[p]['fpr'][sorted_range], base_approach[p]['ppv'][sorted_range])
                plt.plot(x, interpolated_rate / interpolated_rate_base, label='PPV%s /%s'%(title_suffixes[n], title_suffixes[0]), linestyle=':', color=next(colors))

            plt.axhline(y=1., color='dimgrey', linestyle='--')
            plt.grid(b=True, axis='both')
            plt.xlabel('False Positive Rate')
            plt.ylabel('Rate Ratios')
            plt.legend()

            self.pyplot_sanitize_show('Diff Statistics: ' + drawing_title.get_text() + ','.join(str(suffix) for suffix in title_suffixes))
