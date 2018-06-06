#!/usr/bin/env python3

import argparse
import itertools
import os
import pickle
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import root_pandas as rpd

import pdg

try:
    import seaborn as sns

    # Enable and customize default plotting style
    sns.set_style("whitegrid")
except ImportError:
    pass


def pdg_from_name_faulty(particle):
    """Return the pdgCode for a given particle honoring a bug in the float to integer conversion.

    Args:
        particle (str): The name of the particle which should be translated.

    Returns:
        int: The Particle Data Group (PDG) code compatible with the values in ROOT files.

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
        pdg_code (:obj:`float` or :obj:`int`): PDG code compatible with the values in ROOT files.

    Returns:
        str: The name of the particle for the given PDG code with 'none' as value for faulty reconstructions and 'nan' for buggy translations.

    """
    if pdg_code == 0:
        return 'none'

    try:
        return pdg.to_name(int(pdg_code))
    except LookupError:
        return 'nan'


def basf2_Code(particle):
    """Return the pdgCode in a basf2 compatible way with escaped special characters.

    Args:
        particle (str): The name of the particle which should be translated.

    Returns:
        str: Return the escaped pdgCode.

    Raises:
        ValueError: Bogus particle code of the given particle which is neither > 0 nor < 0.

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
        particles (:obj:`list` of :obj:`str`): List of stable particles.
        particles_bar (:obj:`list` of :obj:`str`): List of charged conjugates of stable particles.
        particleIDs (:obj:`dict` of :obj:`str`): Particle IDs according to the generic ID process for stable particles.
        particle_formats (:obj:`dict` of :obj:`str`): Format string in a dictionary of stable and unstable particles and anti-particles.
        particle_base_formats (:obj:`dict` of :obj:`str`): Format string of the base particle for each stable and unstable particles and anti-particles.
        detectors (:obj:`list` of :obj:`str`): List of real detectors.
        pseudo_detectors (:obj:`list` of :obj:`str`): List of pseudo detectors which are composed of real detectors.
        variable_formats (:obj:`dict` of :obj:`str`): Format strings for ROOT variables.
        variable_units (:obj:`dict` of :obj:`str`): Format strings for the unit of ROOT variables.
        detector_weights (:obj:`dict` of :obj:`float`): Relative weights of detectors.
        physical_boundaries (:obj:`set` of :obj:`str`): Queries for ROOT variables which select the data that physically make sense.

    """
    # Base definitions of stable particles and detector data
    particles = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron']
    particles_bar = ['K-', 'pi-', 'e-', 'mu-', 'anti-p-', 'anti-deuteron']
    particles_charge_conjugate = {'K+': 'K-', 'K-': 'K+', 'pi+': 'pi-', 'pi-': 'pi+', 'e+': 'e-', 'e-': 'e+', 'mu+': 'mu-', 'mu-': 'mu+', 'p+': 'anti-p-', 'anti-p-': 'p+', 'deuteron': 'anti-deuteron', 'anti-deuteron': 'deuteron'}
    particleIDs = {'K+': 'kaonID', 'pi+': 'pionID', 'e+': 'electronID', 'mu+': 'muonID', 'p+': 'protonID', 'deuteron': 'deuteronID'}
    particle_formats = {'K+': r'$K^+$', 'K-': r'$K^-$', 'pi+': r'$\pi^+$', 'pi-': r'$\pi^-$', 'e+': r'$e^+$', 'e-': r'$e^-$', 'mu+': r'$\mu^+$', 'mu-': r'$\mu^-$', 'p+': r'$p^+$', 'p-': r'$p^-$', 'anti-p-': r'$\bar{p}^-$', 'deuteron': r'$d$', 'anti-deuteron': r'$\bar{d}$', 'Sigma+': r'$\Sigma^+$', 'Sigma-': r'$\Sigma^-$', 'anti-Sigma+': r'$\bar{\Sigma}^+$', 'anti-Sigma-': r'$\bar{\Sigma}^-$', 'Xi+': r'$\Xi^+$', 'Xi-': r'$\Xi^-$', 'anti-Xi+': r'$\bar{\Xi}^+$', 'anti-Xi-': r'$\bar{\Xi}^-$', 'none': r'$?$', 'nan': r'$NaN$'}
    particle_base_formats = {'K+': r'$K$', 'K-': r'$K$', 'pi+': r'$\pi$', 'pi-': r'$\pi$', 'e+': r'$e$', 'e-': r'$e$', 'mu+': r'$\mu$', 'mu-': r'$\mu$', 'p+': r'$p$', 'p-': r'$p$', 'deuteron': r'$d$', 'Sigma+': r'$\Sigma$', 'Sigma-': r'$\Sigma$', 'Xi+': r'$\Xi$', 'Xi-': r'$\Xi$', 'None': r'$None$', 'nan': r'$NaN$'}
    detectors = ['svd', 'cdc', 'top', 'arich', 'ecl', 'klm']
    pseudo_detectors = ['all', 'default']
    variable_formats = {'p': r'$p$', 'pErr': r'$p_{Err}$', 'phi': r'$\phi$', 'phiErr': r'$\phi_{Err}$', 'pt': r'$p_t$', 'ptErr': r'${p_t}_{Err}$', 'z0': r'$z0$', 'd0': r'$d0$', 'omega': r'$\omega$', 'omegaErr': r'$\omega_{Err}$', 'Theta': r'$\Theta$', 'ThetaErr': r'$\Theta_{Err}$', 'cosTheta': r'$\cos(\Theta)$'}
    variable_units = {'p': r'$\mathrm{GeV/c}$', 'phi': r'$Rad$', 'pt': r'$\mathrm{GeV/c}$', 'z0': r'$?$', 'd0': r'$?$', 'omega': r'$?$', 'Theta': r'$Rad$', 'cosTheta': r'$unitless$'}
    # Use the detector weights to exclude certain detectors, e.g. for debugging purposes
    # Bear in mind that if all likelihoods are calculated correctly this should never improve the result
    detector_weights = {d: 1. for d in detectors + pseudo_detectors}
    # Queries for variables for selecting physically sensible results
    physical_boundaries = {'0.05 < pt < 5.29', 'abs(z0) < 5', 'abs(d0) < 2', 'mcPDG != 0'}

    def __init__(self, pickle_path=None, input_directory=None, output_directory=None, interactive=None, descriptions=None):
        """Initialize and empty ParticleFrame.

        Args:
            input_directory (:obj:`str`, optional): Default input directory for ROOT files for each particle.
            pickle_path (:obj:`str`, optional): Default input filepath for a pickle from which to initialize the class object.
            output_directory (:obj:`str`, optional): Default output directory for data generated using this ParticleFrame.
            interactive (:obj:`bool`, optional): Whether plotting should be done interactively.
            descriptions (:obj:`dict` of :obj:`str`, optional): Descriptions for cutting columns. For each key there shall exist a correspond cutting column.

        Raises:
            ValueError: If given not-none values for both `pickle_path` and `input_directory`.

        """
        self.data = {}
        if pickle_path is not None and input_directory is not None:
            raise ValueError('invalid number of inputs; Received `pickle_path` and `input_directory`; Please decide upon one method for class initialization')
        if input_directory is not None:
            self.read_root(input_directory)
        elif pickle_path is not None:
            self.read_pickle(pickle_path)
        self.output_directory = os.path.join('res', '') if output_directory is None else output_directory
        self.interactive = False if interactive is None else interactive
        self.descriptions = {} if descriptions is None else descriptions

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
        """Read in the particle information contained within the given directory into the current object and drop un-physical values.

        Args:
            input_directory (str): Directory in which the program shall search for ROOT files for each particle

        """
        # Read in all the particle's information into a dictionary of pandas-frames
        self.data = {p: rpd.read_root(os.path.join(input_directory, p + '.root')) for p in self.particles}
        # Clean up the data; Remove obviously un-physical values
        for particle_data in self.data.values():
            for query in self.physical_boundaries:
                particle_data.query(query, inplace=True)

    def read_pickle(self, pickle_path):
        """Read in the particle information from a pickle file.

        Args:
            pickle_path (str): Filepath of the pickle which shall be loaded.

        """
        loaded = pickle.load(open(pickle_path, 'rb'))
        if type(loaded) == list and len(loaded) == 2:
            self.data, self.descriptions = loaded
        else:
            self.data = loaded

    def save(self, pickle_path=None, output_directory=None):
        """Save the current data of the class to a pickle file.

        Args:
            pickle_path (:obj:`str`, optional): Path where to save the pickle file to; Takes precedence when specified; Do not save anything if given '/dev/null'.
            output_directory (:obj:`str`, optional): Directory where to save the pickle file to with class' name as filename; Do not save anything if specifically given '/dev/null' as output directory.

        """
        if pickle_path is None:
            if output_directory is None:
                pickle_path = os.path.join(self.output_directory, self.__class__.__name__ + '.pkl')
            elif output_directory == '/dev/null':
                pickle_path = '/dev/null'
            else:
                pickle_path = os.path.join(output_directory, self.__class__.__name__ + '.pkl')

        if pickle_path != '/dev/null':
            if not os.path.exists(os.path.dirname(pickle_path)):
                print('Creating desired parent directory "%s" for the pickle file "%s"'%(os.path.dirname(pickle_path), pickle_path), file=sys.stderr)
                os.makedirs(os.path.dirname(pickle_path), exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation

            pickle.dump([self.data, self.descriptions], open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    def stats(self, cut_min=0., cut_max=1., ncuts=50, cutting_columns=None):
        """Calculate, print and plot various values from statistics for further analysis and finally return some values.

        Args:
            cut_min (:obj:`float`, optional): Lower bound of the cut (default: 0).
            cut_max (:obj:`float`, optional): Upper bound of the cut (default: 1).
            ncuts (:obj:`int`, optional): Number of cuts to perform on the interval (default: 50). If given 1 the cut will be at 0.5.
            cutting_columns (:obj:`dict` of :obj:`str`, optional): Dictionary which yields a column name for each particle on which basis the various statistics are calculated.

        Returns:
            :obj:`dict` of :obj:`dict` of :obj:`list` of :obj:`float`: A dictionary of dictionaries containing arrays themselfs.

            Each particle has an entry in the dictionary and each particle's dictionary has a dictionary of values from statistics for each cut:

                {
                    'K+': {
                        'tpr': [True Positive Rate for each cut],
                        'fpr': [False Positive Rate for each cut],
                        'tnr': [True Negative Rate for each cut],
                        'ppv': [Positive Predicted Value for each cut],
                        'fdr': [False Discovery Rate for each cut]
                    },
                    ...
                }

        """
        cutting_columns = self.particleIDs if cutting_columns is None else cutting_columns

        stat = {}
        cuts = np.array([0.5]) if ncuts == 1 else np.linspace(cut_min, cut_max, num=ncuts)
        for p, particle_data in self.items():
            stat[p] = {'tpr': np.array([]), 'fpr': np.array([]), 'tnr': np.array([]), 'ppv': np.array([]), 'fdr': np.array([])}
            for cut in cuts:
                positive = np.float64(particle_data[(particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0])
                negative = np.float64(particle_data[(particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))].shape[0])
                true_positive = np.float64(particle_data[((particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] > cut)].shape[0])
                false_positive = np.float64(particle_data[((particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] > cut)].shape[0])
                true_negative = np.float64(particle_data[((particle_data['mcPDG'] != pdg_from_name_faulty(p)) & (particle_data['mcPDG'] != -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] <= cut)].shape[0])
                true_positive_plus_false_positive = np.float64(particle_data[particle_data[cutting_columns[p]] > cut].shape[0])

                stat[p]['tpr'] = np.append(stat[p]['tpr'], [true_positive / positive])
                stat[p]['fpr'] = np.append(stat[p]['fpr'], [false_positive / negative])
                stat[p]['tnr'] = np.append(stat[p]['tnr'], [true_negative / negative])
                stat[p]['ppv'] = np.append(stat[p]['ppv'], [true_positive / true_positive_plus_false_positive])
                stat[p]['fdr'] = np.append(stat[p]['fdr'], [false_positive / true_positive_plus_false_positive])

                if not np.isclose(stat[p]['fpr'][-1]+stat[p]['tnr'][-1], 1, atol=1e-2):
                    print('VALUES INCONSISTENT: ', end='')

                print('Particle %10s: TPR=%6.6f; FPR=%6.6f; TNR=%6.6f; PPV=%6.6f; FDR=%6.6f; cut=%4.4f'%(p, stat[p]['tpr'][-1], stat[p]['fpr'][-1], stat[p]['tnr'][-1], stat[p]['ppv'][-1], stat[p]['fdr'][-1], cut))

        return stat

    def epsilonPID_matrix(self, cut=0.2, cutting_columns=None):
        """Calculate the epsilon_PID matrix for misclassifying particles: rows represent true particles, columns the classification.

        Args:
            cut (:obj:`float`, optional): Position of the cut for the cutting_columns.
            cutting_columns (:obj:`dict` of :obj:`str`, optional): Dictionary which yields a column name for each particle on which the cuts are performed.

        Returns:
            :obj:`numpy.ndarray`: A numpy matrix of epsilon_PID values. The `epsilon_PID[i][j]` value being the probability given it is a particle ''i'' that it will be categorized as particle ''j''.

        """
        cutting_columns = self.particleIDs if cutting_columns is None else cutting_columns

        epsilonPIDs = np.zeros(shape=(len(self.keys()), len(cutting_columns.keys())))
        for i, i_p in enumerate(self.keys()):
            for j, j_p in enumerate(cutting_columns.keys()):
                # The deuterium code is not properly stored in the mcPDG variable, hence the use of `pdg_from_name_faulty()`
                epsilonPIDs[i][j] = np.float64(self[i_p][((self[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) | (self[i_p]['mcPDG'] == -1 * pdg_from_name_faulty(i_p))) & (self[i_p][cutting_columns[j_p]] > cut)].shape[0]) / np.float64(self[i_p][(self[i_p]['mcPDG'] == pdg_from_name_faulty(i_p)) | (self[i_p]['mcPDG'] == -1 * pdg_from_name_faulty(i_p))].shape[0])

        print("epsilon_PID matrix:\n%s"%(epsilonPIDs))
        return np.nan_to_num(epsilonPIDs)

    def mimic_pid(self, detector_weights=None, check=True):
        """Mimic the calculation of the particleIDs and compare them to their value provided by the analysis software.

        Args:
            detector_weights (:obj:`dict` of :obj:`float`): Dictionary of detectors with the weights (default 1.) as values.
            check (:obj:`bool`, optional): Whether to assert the particleIDs if the detector weights are all 1.

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
            priors (:obj:`dict` of :obj:`float`, optional): Dictionary of 'a priori' weights / probabilities (absolute normalization irrelevant) of detecting a given particle.
            detector (:obj:`str`, optional): Name of the detector which should be inserted into the column string.
            mc_best (:obj:`bool`, optional): Boolean specifying whether to use the Monte Carlo data for calculating the a prior probabilities.

        Returns:
            cutting_columns (:obj:`dict` of :obj:`str`): A dictionary containing the name of each column by particle which shall be used for cuts.

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
        """Compute probabilities for particle hypothesis keeping the `hold` ROOT variable fixed using a Bayesian approach.

        Args:
            holdings (:obj:`dict` of :obj:`str`, optional): List of ROOT variables on which the 'a prior' probability shall depend on.
            nbins (:obj:`int`, optional): Number of bins to use for the `hold` variable when calculating probabilities.
            detector (:obj:`str`, optional): Name of the detector to be used for pidLogLikelihood extraction.
            mc_best (:obj:`bool`, optional): Boolean specifying whether to use the Monte Carlo data for prior probabilities or an iterative approach.
            niterations (:obj:`int`, optional): Number of iterations for the converging approach.
            norm (:obj:`str`, optional): Particle by which abundance to norm the a priori probabilities.
            whis (:obj:`float`, optional): Whiskers, scale of the Inter Quartile Range (IQR) for outlier exclusion.

        Returns:
            cutting_columns (:obj:`dict` of :obj:`str`): A dictionary containing the name of each column by particle which shall be used for cuts.
            category_columns (:obj:`dict` of :obj:`str`): A dictionary of entries for each `hold` with the name of the column in each dataframe which holds the category for bin selection.
            intervals (:obj:`dict` of :obj:`list` of :obj:`float`): A dictionary of entries for each `hold` containing an array of interval boundaries for every bin.
            iteration_priors (:obj:`dict` of :obj:`dict` of :obj:`list` of :obj:`float`): A dictionary for each dataset of particle dictionaries containing arrays of priors for each iteration.

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
            cutting_columns (:obj:`dict` of :obj:`str`): Columns by particles where to find the maximum

        Returns:
            cutting_columns_isMax (:obj:`dict` of :obj:`str`): Columns by particle containing ones for maximal values

        """
        for particle_data in self.values():
            max_columns = particle_data[list(cutting_columns.values())].idxmax(axis=1)
            cutting_columns_isMax = {k: v + '_isMax' for k, v in cutting_columns.items()}

            for p in cutting_columns.keys():
                particle_data[cutting_columns_isMax[p]] = np.where(max_columns == cutting_columns[p], 1, 0)

        return cutting_columns_isMax

    def pyplot_sanitize_show(self, title, savefig_prefix='', savefig_suffix='', suptitle=False, format='pdf', bbox_inches='tight', output_directory=None, interactive=None, **kwargs):
        """Show and save the current figure to a configurable location and sanitize its name.

        Save the plot currently handled via `matplotlib.pyplot` without the title but an appropriate name, then plot the title and display the plot if run interactively.

        Args:
            title (str): Title of the plot and baseline for the name of the file.
            savefig_prefix (:obj:`str`, optional): Title prefix which gets prepended to the filename of the plot but is not displayed otherwise.
            savefig_suffix (:obj:`str`, optional): Title suffix which gets appended to the filename of the plot but is not displayed otherwise.
            suptitle (:obj:`bool`, optional): Whether the title is intended as a super title or as a regular one.
            format (:obj:`str`, optional): Format in which to save the plot; Its value is also appended to the filename.
            bbox_inches (:obj:`str`, optional): Bbox in inches; If 'tight' figure out the best suitable values.
            output_directory (:obj:`str`, optional): Output directory; Defaulting to the ParticleFrame's `self.output_directory`, do not save anything if given '/dev/null' as output.
            interactive (:obj:`bool`, optional): Whether to run interactive or not; Defaulting to the ParticleFrame's `self.interactive`.
            **kwargs: Any keyword arguments valid for `matplotlib.pyplot.savefig()`.

        """
        output_directory = self.output_directory if output_directory is None else output_directory
        interactive = self.interactive if interactive is None else interactive

        if output_directory != '/dev/null':
            if not os.path.exists(output_directory):
                print('Creating desired output directory "%s"'%(output_directory), file=sys.stderr)
                os.makedirs(output_directory, exist_ok=True) # Prevent race conditions by not failing in case of intermediate dir creation

            sanitized_title = re.sub('[\\\\$_^{}]', '', savefig_prefix + title + savefig_suffix)
            plt.savefig(os.path.join(output_directory, sanitized_title + '.' + format), bbox_inches=bbox_inches, format=format, **kwargs)

        if suptitle:
            plt.suptitle(title)
        else:
            plt.title(title)

        if interactive:
            plt.show(block=False)
        else:
            plt.close()

    def plot_stats_by_particle(self, stat, particles_of_interest=None, **kwargs):
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        for p in particles_of_interest:
            plt.figure()
            plt.plot(stat[p]['fpr'], stat[p]['tpr'], label='ROC')
            # Due to the fact that FPR + TNR = 1 the plot will simply show a straight line; Use for debugging only
            # plt.plot(stat[p]['fpr'], stat[p]['tnr'], label='True Negative Rate')
            plt.plot(stat[p]['fpr'], stat[p]['ppv'], label='PPV')
            plt.xlabel('False Positive Rate')
            plt.xlim(-0.05, 1.05)
            plt.ylabel('Particle Rates')
            plt.ylim(-0.05, 1.05)
            plt.legend()
            self.pyplot_sanitize_show('%s Identification'%(self.particle_base_formats[p]), **kwargs)

    def plot_epsilonPIDs(self, epsilonPIDs_approach, **kwargs):
        plt.figure()
        plt.imshow(epsilonPIDs_approach, cmap='viridis', vmin=0., vmax=1.)
        for (j, i), label in np.ndenumerate(epsilonPIDs_approach):
            plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small', color=str(np.piecewise(label, [label < 0.5, label >= 0.5], [1, 0])))
        plt.grid(b=False, axis='both')
        plt.xlabel('Predicted Particle')
        plt.xticks(range(len(ParticleFrame.particles)), [ParticleFrame.particle_base_formats[p] for p in ParticleFrame.particles])
        plt.ylabel('True Particle')
        plt.yticks(range(len(ParticleFrame.particles)), [ParticleFrame.particle_base_formats[p] for p in ParticleFrame.particles])
        plt.colorbar()
        self.pyplot_sanitize_show(**kwargs)

    def plot_neyman_pearson(self, nbins=10, cutting_columns=None, title_suffix='', particles_of_interest=None, bar_particles=False, binning_method=None, hold='pt', n_hold_cuts=3, **kwargs):
        cutting_columns = self.particleIDs if cutting_columns is None else cutting_columns
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        # NOTE: This is one of the few place where we differentiate between particle and anti-particle
        charge_queries = ['charge > 0', 'charge < 0', 'charge != 0'] if bar_particles else ['charge != 0']
        for query, p in itertools.product(charge_queries, particles_of_interest):
            particle_data_charged = self[p].query(query)
            if query == 'charge > 0':
                current_format = self.particle_formats[p]
            elif query == 'charge < 0':
                current_format = self.particle_formats[self.particles_charge_conjugate[p]]
            else:
                current_format = self.particle_base_formats[p]

            if binning_method == 'qcut':
                categories, category_intervals = pd.qcut(particle_data_charged[hold], q=n_hold_cuts, labels=range(n_hold_cuts), retbins=True)
                title_appendix = r' for equal size %s bins'%(self.variable_formats[hold])
            elif binning_method == 'cut':
                categories, category_intervals = pd.cut(particle_data_charged[hold], bins=n_hold_cuts, labels=range(n_hold_cuts), retbins=True)
                title_appendix = r' for equal width %s bins'%(self.variable_formats[hold])
            else:
                categories = np.zeros(particle_data_charged.shape[0])
                category_intervals = [None, None]
                title_appendix = ''

            if len(category_intervals) > 2:
                plt.figure(figsize=(10, 3))
            else:
                plt.figure()
            for i in range(len(category_intervals) - 1):
                selection = (categories == i)
                if len(category_intervals) > 2:
                    plt.subplot(1, len(category_intervals)-1, i+1)
                    plt.title('(%.2f < %s < %.2f)'%(category_intervals[i], self.variable_formats[hold], category_intervals[i+1]), fontsize=10)
                plt.xlabel(r'$\mathcal{LR}($%s$)$'%(current_format))

                if i == 0:
                    plt.ylabel('Relative Abundance')
                else:
                    plt.setp(plt.gca().get_yticklabels(), visible=False)
                plt.xlabel(r'$\mathcal{LR}($%s$)$'%(current_format))

                likelihood_ratio_bins, intervals = pd.cut(particle_data_charged[selection][cutting_columns[p]], nbins, labels=range(nbins), retbins=True)
                abundance_ratio = np.zeros(nbins)
                y_err = np.zeros(nbins)
                for i in range(nbins):
                    particle_data_bin = particle_data_charged[selection][likelihood_ratio_bins == i]
                    if query == 'charge > 0':
                        numerator = particle_data_bin[particle_data_bin['mcPDG'] == pdg_from_name_faulty(p)].shape[0]
                    elif query == 'charge < 0':
                        numerator = particle_data_bin[particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p)].shape[0]
                    else:
                        numerator = particle_data_bin[(particle_data_bin['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0]

                    denominator = np.array([particle_data_bin[(particle_data_bin['mcPDG'] == pdg_from_name_faulty(p_2)) | (particle_data_bin['mcPDG'] == -1 * pdg_from_name_faulty(p_2))].shape[0] for p_2 in self.particles]).sum()

                    abundance_ratio[i] = numerator / denominator
                    y_err[i] = np.sqrt(abundance_ratio[i] * (1 - abundance_ratio[i]) / denominator)

                interval_centers = np.array([np.mean(intervals[i:i+2]) for i in range(len(intervals)-1)])
                # Indicate the width of the bins via x errorbars
                x_err = np.array([intervals[i] - intervals[i-1] for i in range(1, len(intervals))]) / 2.

                plt.ylim(-0.05, 1.05)
                plt.xlim(-0.05, 1.05)
                plt.errorbar(interval_centers, abundance_ratio, xerr=x_err, yerr=y_err, capsize=0., elinewidth=1, marker='o', markersize=4, markeredgewidth=1, markerfacecolor='None', linestyle='--', linewidth=0.1)

            suptitle = True if len(category_intervals) > 2 else False
            self.pyplot_sanitize_show('Relative %s Abundance in Likelihood Ratio Bins%s%s'%(current_format, title_suffix, title_appendix), suptitle=suptitle, **kwargs)

    def plot_diff_epsilonPIDs(self, epsilonPIDs_approaches=[], title_suffixes=[], title_epsilonPIDs='', **kwargs):
        if len(epsilonPIDs_approaches) >= 0 and len(epsilonPIDs_approaches) != len(title_suffixes):
            raise ValueError('epsilonPIDs_approaches array must be of same length as the title_suffixes array')

        fig, _ = plt.subplots(nrows=len(epsilonPIDs_approaches), ncols=1)
        for n in range(len(epsilonPIDs_approaches)):
            plt.subplot(1, len(epsilonPIDs_approaches), n+1)
            plt.imshow(epsilonPIDs_approaches[n], cmap='viridis', vmin=0., vmax=1.)
            for (j, i), label in np.ndenumerate(epsilonPIDs_approaches[n]):
                plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small', color=str(np.piecewise(label, [label < 0.5, label >= 0.5], [1, 0])))
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
        self.pyplot_sanitize_show(title_epsilonPIDs, suptitle=True, **kwargs)

    def plot_diff_stats(self, stats_approaches=[], title_suffixes=[], x_axis=('fpr', 'False Positive Rate'), y_multi_axis=['tpr', 'ppv'], x_lim=(-0.05, 1.05), y_lim=(-0.05, 1.05), particles_of_interest=None, ninterpolations=100, **kwargs):
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        if len(stats_approaches) >= 0 and len(stats_approaches) != len(title_suffixes):
            raise ValueError('stats_approaches array must be of same length as the title_suffixes array')

        for p in particles_of_interest:
            plt.figure()
            grid = plt.GridSpec(3, 1, hspace=0.1)
            colors = iter(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

            main_ax = plt.subplot(grid[:2, 0])
            for n, approach in enumerate(stats_approaches):
                markers = ('o', '^') if len(approach[p][x_axis[0]]) == 1 else (None, None)
                drawing = plt.plot(approach[p][x_axis[0]], approach[p][y_multi_axis[0]], marker=markers[0], label=y_multi_axis[0].upper() + title_suffixes[n], color=next(colors))
                for y_axis in y_multi_axis[1:]:
                    plt.plot(approach[p][x_axis[0]], approach[p][y_axis], marker=markers[1], label=y_axis.upper() + title_suffixes[n], linestyle=':', color=drawing[0].get_color())

            plt.setp(main_ax.get_xticklabels(), visible=False)
            plt.ylabel('Particle Rates')
            plt.ylim(y_lim)
            plt.legend()

            plt.subplot(grid[2, 0], sharex=main_ax)
            base_approach = stats_approaches[0]
            # Numpy expects values sorted by x
            sorted_base_range = np.argsort(base_approach[p][x_axis[0]])
            for n, approach in enumerate(stats_approaches[1:], 1):
                # Skip interpolation if the approach contains only one point
                if len(approach[p][x_axis[0]]) == 1:
                    continue

                sorted_approach_range = np.argsort(approach[p][x_axis[0]])
                x_min = max(base_approach[p][x_axis[0]][sorted_base_range][1], approach[p][x_axis[0]][sorted_approach_range][1]) # Skip the first FPR value (probably zero)
                x_max = min(base_approach[p][x_axis[0]][sorted_base_range][-1], approach[p][x_axis[0]][sorted_approach_range][-1])
                x = np.linspace(x_min, x_max, ninterpolations)

                for i, y_axis in enumerate(y_multi_axis):
                    linestyle = None if i == 0 else ':'
                    interpolated_rate = np.interp(x, approach[p][x_axis[0]][sorted_approach_range], approach[p][y_axis][sorted_approach_range])
                    interpolated_rate_base = np.interp(x, base_approach[p][x_axis[0]][sorted_base_range], base_approach[p][y_axis][sorted_base_range])
                    plt.plot(x, interpolated_rate / interpolated_rate_base, label='%s%s /%s'%(y_axis.upper(), title_suffixes[n], title_suffixes[0]), linestyle=linestyle, color=next(colors))

            plt.axhline(y=1., color='dimgrey', linestyle='--')
            plt.grid(b=True, axis='both')
            plt.xlabel(x_axis[1])
            plt.xlim(x_lim)
            plt.ylabel('Rate Ratios')
            plt.legend()

            self.pyplot_sanitize_show('%s Identification'%(self.particle_base_formats[p]), suptitle=True, **kwargs)

    def plot_diff_abundance(self, cutting_columns_approaches=[], title_suffixes=[], particles_of_interest=None, norm='K+', cut=0.5, **kwargs):
        particles_of_interest = self.particles if particles_of_interest is None else particles_of_interest

        particle_data = self[norm]

        plt.figure()
        plt.grid(b=False, axis='x')

        true_abundance = np.array([particle_data[(particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))].shape[0] for p in particles_of_interest])
        sorted_range = np.argsort(true_abundance)[::-1]
        plt.errorbar(range(len(particles_of_interest)), true_abundance[sorted_range], xerr=0.5, marker='None', linestyle='None', label='truth')

        for i, cutting_columns in enumerate(cutting_columns_approaches):
            # Abundances might vary for different datasets due to some preliminary mass hypothesis being applied on reconstruction
            abundance_correct = np.array([particle_data[((particle_data['mcPDG'] == pdg_from_name_faulty(p)) | (particle_data['mcPDG'] == -1 * pdg_from_name_faulty(p))) & (particle_data[cutting_columns[p]] > cut)].shape[0] for p in particles_of_interest])
            abundance = np.array([particle_data[particle_data[cutting_columns[p]] > cut].shape[0] for p in particles_of_interest])

            drawing = plt.errorbar(range(len(particles_of_interest)), abundance_correct[sorted_range], xerr=0.5, elinewidth=0.8, marker='None', linestyle='None')
            plt.errorbar(range(len(particles_of_interest)), abundance[sorted_range], xerr=0.5, marker='None', linestyle='None', color=drawing[0].get_color(), label=title_suffixes[i].lstrip() + ' (absolute, true)')

        plt.legend()
        sorted_particles = np.array(particles_of_interest)[sorted_range]
        plt.xticks(range(len(particles_of_interest)), [self.particle_base_formats[p] for p in sorted_particles])
        self.pyplot_sanitize_show('Particle Abundances in the %s-Data'%(ParticleFrame.particle_formats[norm]), **kwargs)
