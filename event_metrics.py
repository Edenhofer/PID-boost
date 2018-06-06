#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions = 1   # This option has to bet set prior to importing argparse

import argparse
import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np

import lib

try:
    import argcomplete
except ImportError:
    pass


ParticleFrame = lib.ParticleFrame

# Assemble the allowed command line options
parser = argparse.ArgumentParser(description='Calculating and visualizing metrics.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group_action = parser.add_argument_group('actions', 'Parameters which induce some kind of calculations and or visualizations')
group_opt = parser.add_argument_group('sub-options', 'Parameters which make only sense to use in combination with an action and which possibly alters their behavior')
group_util = parser.add_argument_group('utility options', 'Parameters for altering the behavior of the program\'s input-output handling')
group_action.add_argument('--stats', dest='run_stats', action='store_true', default=False,
                    help='Visualize some general purpose statistics of the dataset')
group_action.add_argument('--pid', dest='run_pid', action='store_true', default=False,
                    help='Print out and visualize some statistics and the epsilonPID-matrix for the default particle ID cut')
group_action.add_argument('--pidProbability', dest='run_pidProbability', nargs='?', action='store', choices=['plain', 'vertex', 'primary'], default=None, const='plain',
                    help='Print out and visualize some statistics and the epsilonPID-matrix for the pidProbability cut; Optionally slice the data to fulfill certain vertex requirements')
group_action.add_argument('--pidProbability-motivation', dest='run_pidProbability_motivation', action='store_true', default=False,
                    help='Visualize the goodness of detector components in different ranges')
group_action.add_argument('--mimic-pid', dest='run_mimic_pid', action='store_true', default=False,
                    help='Mimic the calculation of the particle IDs using likelihoods')
group_action.add_argument('--bayes', dest='run_bayes', action='store_true', default=False,
                    help='Calculate an accumulated probability for particle hypothesis using Bayes')
group_action.add_argument('--diff', dest='diff_methods', nargs='+', action='store', choices=['pid', 'flat_bayes', 'simple_bayes', 'pidProbability', 'univariate_bayes', *['univariate_bayes_by_' + v for v in ParticleFrame.variable_formats.keys()], 'multivariate_bayes', 'nn'], default=[],
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
group_opt.add_argument('--bar-particles', dest='bar_particles', action='store_true', default=False,
                    help='Whether to additionally differentiate by charge when performing the test based on the Neyman-Pearson lemma')
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
group_opt.add_argument('--particles-of-interest', dest='particles_of_interest', nargs='+', action='store', choices=ParticleFrame.particles, default=ParticleFrame.particles,
                    help='List of particles which shall be analysed')
group_opt.add_argument('--whis', dest='whis', nargs='?', action='store', type=float, default=None,
                    help='Whiskers with which the IQR will be IQR')
group_util.add_argument('-i', '--input', dest='input_directory', action='store', default='./',
                    help='Directory in which the program shall search for ROOT files for each particle')
group_util.add_argument('--input-pickle', dest='input_pickle', action='store', default=None,
                    help='Pickle file path containing a ParticleFrame object which shall be read in instead of ROOT files; Takes precedence when specified')
group_util.add_argument('-o', '--output', dest='output_directory', action='store', default='./res/',
                    help='Directory for the generated output (mainly plots); Skip saving plots if given \'/dev/null\'.')
group_util.add_argument('--interactive', dest='interactive', action='store_true', default=True,
                    help='Run interactively, i.e. show plots')
group_util.add_argument('--non-interactive', dest='interactive', action='store_false', default=True,
                    help='Run non-interactively and hence unattended, i.e. show no plots')

try:
    argcomplete.autocomplete(parser)
except NameError:
    pass


args = parser.parse_args()

# Read in all the particle's information into a dictionary of pandas-frames
input_directory = args.input_directory
input_pickle = args.input_pickle
interactive = args.interactive
output_directory = args.output_directory
if input_pickle:
    data = ParticleFrame(pickle_path=input_pickle, output_directory=output_directory, interactive=interactive)
else:
    data = ParticleFrame(input_directory=input_directory, output_directory=output_directory, interactive=interactive)

if args.run_stats:
    particles_of_interest = args.particles_of_interest

    # Abundances might vary due to some preliminary mass hypothesis being applied on reconstruction
    # ~~Ignore this for now and just plot the dataset belonging to the `norm` particle~~ --> Do this for every particle in particle_of_interest
    for p in particles_of_interest:
        particle_data = data[p]

        # NOTE: This is one of the few place where we differentiate between particle and anti-particle
        unique_particles = np.unique(particle_data['mcPDG'].values)
        true_abundance = np.array([particle_data[particle_data['mcPDG'] == code].shape[0] for code in unique_particles])
        sorted_range = np.argsort(true_abundance)
        true_abundance = true_abundance[sorted_range][::-1]
        unique_particles = unique_particles[sorted_range][::-1]

        plt.figure()
        plt.grid(b=False, axis='x')
        plt.errorbar(range(len(unique_particles)), true_abundance, xerr=0.5, fmt='o')
        plt.xticks(range(len(unique_particles)), [ParticleFrame.particle_formats[lib.pdg_to_name_faulty(k)] for k in unique_particles])
        data.pyplot_sanitize_show('True Particle Abundances in the %s-Data'%(ParticleFrame.particle_formats[p]), savefig_prefix='General Purpose Statistics: ')

if args.run_pid:
    cut = args.cut
    exclusive_cut = args.exclusive_cut

    particles_of_interest = args.particles_of_interest

    c = ParticleFrame.particleIDs
    data.plot_stats_by_particle(data.stats(cutting_columns=c), particles_of_interest=particles_of_interest, savefig_prefix='Particle ID Approach: ')
    c = data.add_isMax_column(c) if exclusive_cut else c
    epsilonPIDs = data.epsilonPID_matrix(cutting_columns=c, cut=cut)
    if exclusive_cut:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    data.plot_epsilonPIDs(epsilonPIDs, title=drawing_title, savefig_prefix='Particle ID Approach: ')

if args.run_pidProbability:
    # This call is special in the sense that it is the currently the best way to identify particle, hence it got some special attention and does some more stuff
    slicing_method = args.run_pidProbability
    norm = args.norm

    cut = args.cut
    nbins = args.nbins
    holdings = args.holdings
    exclusive_cut = args.exclusive_cut
    bar_particles = args.bar_particles

    particles_of_interest = args.particles_of_interest

    detectors_of_interest = ['all', 'cdc']

    if slicing_method == 'vertex':
        print('Description of raw %s-Data:\n'%(norm), data[norm][['d0', 'z0', 'mcDX', 'mcDY', 'mcDZ']].describe())

        plt.figure()
        for i in ['mcDX', 'mcDY', 'mcDZ']:
            plt.hist(data[norm][(data[norm]['mcPDG'] == lib.pdg_from_name_faulty(norm)) | (data[norm]['mcPDG'] == -1 * lib.pdg_from_name_faulty(norm))][i], histtype='step', label=i)
        plt.legend()
        data.pyplot_sanitize_show('Decay Vertex of matched MC %s'%(ParticleFrame.particle_base_formats[norm]), savefig_prefix='General Purpose Statistics: ')

        special_slice = ParticleFrame(output_directory=output_directory, interactive=interactive)
        special_slice.data = data.data.copy()
        for particle_data in special_slice.data.values():
            # The unit of mcDX, mcDY and mcDZ is not the same as the one of d0 and z0
            particle_data.query('(mcDX**2 + mcDY**2 + mcDZ**2) < 120000', inplace=True)
        print('Description of sliced %s-Data:\n'%(norm), special_slice[norm][['d0', 'z0', 'mcDX', 'mcDY', 'mcDZ']].describe())
        savefig_prefix='pidProbability Approach (tight Vertex): '
    if slicing_method == 'primary':
        special_slice = ParticleFrame(output_directory=output_directory, interactive=interactive)
        special_slice.data = data.data.copy()
        for particle_data in special_slice.data.values():
            particle_data.query('isPrimarySignal == 1', inplace=True)
        savefig_prefix='pidProbability Approach (primary Signal): '
    else:
        special_slice = data
        savefig_prefix='pidProbability Approach: '

    c = {p: 'pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + detectors_of_interest[0] + '__bc' for p in ParticleFrame.particles}
    special_slice.plot_stats_by_particle(special_slice.stats(cutting_columns=c), particles_of_interest=particles_of_interest, savefig_prefix=savefig_prefix)
    c = special_slice.add_isMax_column(c) if exclusive_cut else c
    epsilonPIDs = special_slice.epsilonPID_matrix(cutting_columns=c, cut=cut)
    if exclusive_cut:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    special_slice.plot_epsilonPIDs(epsilonPIDs, title=drawing_title, savefig_prefix=savefig_prefix)

    for d, hold, binning_method in itertools.product(detectors_of_interest, holdings, ['qcut', 'cut']):
        c = {p: 'pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc' for p in ParticleFrame.particles}
        # Ignore the wish of the user and never plot those visuals for bar_particles as this would create a lot of figures
        special_slice.plot_neyman_pearson(cutting_columns=c, title_suffix=' for %s detector'%(d.upper()), particles_of_interest=particles_of_interest, bar_particles=False, binning_method=binning_method, hold=hold, savefig_prefix=savefig_prefix)
    # NOTE: In contrast to the other methods we differentiate by detector for the `pidProbability` variable
    for d in ParticleFrame.detectors + ParticleFrame.pseudo_detectors:
        c = {p: 'pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc' for p in ParticleFrame.particles}
        special_slice.plot_neyman_pearson(cutting_columns=c, title_suffix=' for %s detector'%(d.upper()), particles_of_interest=particles_of_interest, bar_particles=bar_particles, savefig_prefix=savefig_prefix)

if args.run_pidProbability_motivation:
    whis = args.whis
    nbins = args.nbins
    particles_of_interest = args.particles_of_interest
    holdings = args.holdings

    detectors = ['all']
    threshold = 0.05

    for p, d in itertools.product(particles_of_interest, detectors):
        color_column = 'pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc'
        particle_data_above_threshold = data[p].query(color_column + ' > ' + str(threshold))
        current_format = data.particle_base_formats[p]

        selection = np.ones(particle_data_above_threshold.shape[0], dtype=bool)
        if whis:
            for hold in holdings:
                q75, q25 = np.percentile(particle_data_above_threshold[hold], [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - (iqr * whis)
                upper_bound = q75 + (iqr * whis)
                selection = selection & (particle_data_above_threshold[hold] > lower_bound) & (particle_data_above_threshold[hold] < upper_bound)

        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

        main_ax = plt.subplot(grid[:-1, 1:])
        plt.scatter(particle_data_above_threshold[selection][holdings[0]], particle_data_above_threshold[selection][holdings[1]], c=particle_data_above_threshold[selection][color_column], cmap='viridis', s=5., alpha=.1)
        plt.setp(main_ax.get_xticklabels(), visible=False)
        plt.setp(main_ax.get_yticklabels(), visible=False)

        plt.subplot(grid[-1, 1:], sharex=main_ax)
        plt.hist(particle_data_above_threshold[selection][holdings[0]], nbins, histtype='step', orientation='vertical')
        plt.gca().invert_yaxis()
        plt.xlabel(ParticleFrame.variable_formats[holdings[0]] + ' (' + ParticleFrame.variable_units[holdings[0]] + ')')
        plt.subplot(grid[:-1, 0], sharey=main_ax)
        plt.hist(particle_data_above_threshold[selection][holdings[1]], nbins, histtype='step', orientation='horizontal')
        plt.gca().invert_xaxis()
        plt.ylabel(ParticleFrame.variable_formats[holdings[1]] + ' (' + ParticleFrame.variable_units[holdings[1]] + ')')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.20, 0.05, 0.6])
        cbar = plt.colorbar(cax=cbar_ax)
        cbar.set_alpha(1.)
        cbar.draw_all()

        data.pyplot_sanitize_show(current_format + ' pidPropability multi-axes Histogram of ' + ', '.join(format(hold) for hold in holdings) + ' above ' + str(threshold) + ' Threshold', savefig_prefix='pidProbability Approach: ', suptitle=True, format='png')

if args.run_mimic_pid:
    data.mimic_pid()

if args.run_bayes:
    mc_best = args.mc_best
    cut = args.cut
    exclusive_cut = args.exclusive_cut
    bar_particles = args.bar_particles
    particles_of_interest = args.particles_of_interest

    c = data.bayes(mc_best=mc_best)
    data.plot_stats_by_particle(data.stats(cutting_columns=c), particles_of_interest=particles_of_interest, savefig_prefix='Bayesian Approach: ')
    c = data.add_isMax_column(c) if exclusive_cut else c
    epsilonPIDs = data.epsilonPID_matrix(cutting_columns=c, cut=cut)
    if exclusive_cut:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    data.plot_epsilonPIDs(epsilonPIDs, title=drawing_title, savefig_prefix='Bayesian Approach: ')
    suf = ' via simple Bayes' if mc_best else ' via flat Bayes'
    data.plot_neyman_pearson(cutting_columns=c, title_suffix=suf, particles_of_interest=particles_of_interest, bar_particles=bar_particles, savefig_prefix='Bayesian Approach: ')

if args.diff_methods:
    methods = args.diff_methods
    assert (len(methods) >= 1), 'Specify at least one methods'

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

    detector = 'all'

    epsilonPIDs_approaches = []
    stats_approaches = []
    cutting_columns_approaches = []
    title_suffixes = []
    savefig_suffix = ''
    for m in methods:
        current_ncuts = ncuts

        if m == 'pid':
            c = ParticleFrame.particleIDs
            t = ' via PID'
        elif m == 'flat_bayes':
            c = data.bayes()
            t = ' via flat Bayes'
        elif m == 'pidProbability':
            # Consistency check for flat_bayes, as both should yield the same results (mathematically)
            c = {p: 'pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + detector + '__bc' for p in ParticleFrame.particles}
            t = ' via pidProbability'
        elif m == 'simple_bayes':
            c = data.bayes(mc_best=True)
            t = ' via simple Bayes'
        elif m == 'univariate_bayes':
            c = data.multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            t = ' via univariate Bayes'
        elif m in ['univariate_bayes_by_' + v for v in ParticleFrame.variable_formats.keys()]:
            explicit_hold = m.replace('univariate_bayes_by_', '')
            c = data.multivariate_bayes(holdings=[explicit_hold], whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            t = ' by ' + ParticleFrame.variable_formats[explicit_hold]
        elif m == 'multivariate_bayes':
            c = data.multivariate_bayes(holdings=holdings, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)[0]
            t = ' by ' + ' & '.join([ParticleFrame.variable_formats[h] for h in holdings])
        elif m == 'nn':
            # In sharp contrast to all the previous approaches this one assumes the columns are already present and will not generate them beforehand!
            c = {k: 'nn_' + v for k, v in ParticleFrame.particleIDs.items()}
            current_ncuts = 1
            t = ' via NN'
        else:
            raise ValueError('received unknown method "%s"'%(m))

        if m in data.descriptions:
            s = ' ' + data.descriptions[m]
        else:
            s = t

        title_suffixes += [t]
        savefig_suffix = s if savefig_suffix == '' else ','.join([savefig_suffix, s])
        c_choice = data.add_isMax_column(c) if exclusive_cut else c
        cutting_columns_approaches += [c_choice]
        epsilonPIDs_approaches += [data.epsilonPID_matrix(cutting_columns=c_choice, cut=cut)]
        stats_approaches += [data.stats(cutting_columns=c, ncuts=current_ncuts)]

    if exclusive_cut:
        title_epsilonPIDs = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        title_epsilonPIDs = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    data.plot_diff_epsilonPIDs(epsilonPIDs_approaches=epsilonPIDs_approaches, title_suffixes=title_suffixes, title_epsilonPIDs=title_epsilonPIDs, savefig_prefix='Diff Heatmap: ', savefig_suffix=savefig_suffix)
    data.plot_diff_stats(stats_approaches=stats_approaches, title_suffixes=title_suffixes, particles_of_interest=particles_of_interest, savefig_prefix='Diff Statistics: ', savefig_suffix=savefig_suffix)
    data.plot_diff_stats(stats_approaches=stats_approaches, x_axis=('tpr', 'Efficiency'), y_multi_axis=[('ppv', 'Purity')], x_lim=(None, None), title_suffixes=title_suffixes, particles_of_interest=particles_of_interest, savefig_prefix='Diff Statistics: ', savefig_suffix=' TPR over PPV' +  savefig_suffix)
    data.plot_diff_stats(stats_approaches=stats_approaches, x_axis=('tpr', 'Efficiency'), y_multi_axis=[('ppv', 'Purity')], x_lim=(None, None), title_suffixes=title_suffixes, particles_of_interest=particles_of_interest, savefig_prefix='Diff Statistics: ', savefig_suffix=' TPR over PPV' + savefig_suffix, ratios=False)
    data.plot_diff_abundance(cutting_columns_approaches=cutting_columns_approaches, title_suffixes=title_suffixes, particles_of_interest=particles_of_interest, savefig_prefix='Diff Abundances: ', savefig_suffix=savefig_suffix)

if args.run_univariate_bayes:
    cut = args.cut
    bar_particles = args.bar_particles

    hold = args.hold
    whis = args.whis
    niterations = args.niterations
    nbins = args.nbins
    norm = args.norm
    mc_best = args.mc_best
    exclusive_cut = args.exclusive_cut
    cutting_columns, category_columns, intervals, _ = data.multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals[hold].items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals[hold].items()}

    particles_of_interest = args.particles_of_interest

    plt.figure()
    for p in particles_of_interest:
        assumed_abundance = np.array([data[p][((data[p]['mcPDG'] == lib.pdg_from_name_faulty(p)) | (data[p]['mcPDG'] == -1 * lib.pdg_from_name_faulty(p))) & (data[p][category_columns[hold]] == it) & (data[p][cutting_columns[p]] > cut)].shape[0] for it in range(nbins)])
        actual_abundance = np.array([data[p][((data[p]['mcPDG'] == lib.pdg_from_name_faulty(p)) | (data[p]['mcPDG'] == -1 * lib.pdg_from_name_faulty(p))) & (data[p][category_columns[hold]] == it)].shape[0] for it in range(nbins)])
        plt.errorbar(interval_centers[p], assumed_abundance / actual_abundance, xerr=interval_widths[p], label='%s'%(ParticleFrame.particle_base_formats[p]), fmt='o')

    plt.xlabel(ParticleFrame.variable_formats[hold] + ' (' + ParticleFrame.variable_units[hold] + ')')
    plt.ylabel('True Positive Rate')
    plt.legend()
    data.pyplot_sanitize_show('True Positive Rate for a Cut at %.2f'%(cut), savefig_prefix='Univariate Bayesian Approach: ')

    c = data.add_isMax_column(cutting_columns) if exclusive_cut else cutting_columns
    epsilonPIDs = data.epsilonPID_matrix(cutting_columns=c, cut=cut)
    if exclusive_cut:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    data.plot_epsilonPIDs(epsilonPIDs, title=drawing_title, savefig_prefix='Univariate Bayesian Approach: ')

    data.plot_stats_by_particle(data.stats(cutting_columns=cutting_columns), particles_of_interest=particles_of_interest, savefig_prefix='Univariate Bayesian Approach: ')
    data.plot_neyman_pearson(cutting_columns=cutting_columns, title_suffix=' by ' + ParticleFrame.variable_formats[hold], particles_of_interest=particles_of_interest, bar_particles=bar_particles, savefig_prefix='Univariate Bayesian Approach: ')

if args.run_univariate_bayes_priors:
    particles_of_interest = args.particles_of_interest
    hold = args.hold
    whis = args.whis
    nbins = args.nbins
    niterations = args.niterations
    norm = args.norm

    iteration_priors_viaIter = data.multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=False, niterations=niterations, nbins=nbins)[-1]
    intervals, iteration_priors_viaBest = data.multivariate_bayes(holdings=[hold], whis=whis, norm=norm, mc_best=True, nbins=nbins)[-2:]
    interval_centers = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals[hold].items()}
    interval_widths = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals[hold].items()}

    for p in particles_of_interest:
        plt.figure()
        plt.errorbar(interval_centers[p], iteration_priors_viaBest[norm][p][-1], xerr=interval_widths[p], label='Truth', fmt='*')
        for n in range(niterations):
            plt.errorbar(interval_centers[p], iteration_priors_viaIter[norm][p][n], xerr=interval_widths[p], label='Iteration %d'%(n+1), fmt='o')

        plt.xlabel(ParticleFrame.variable_formats[hold] + ' (' + ParticleFrame.variable_units[hold] + ')')
        plt.ylabel('Relative Abundance')
        plt.legend()
        data.pyplot_sanitize_show('%s Spectra Ratios Relative to %s'%(ParticleFrame.particle_base_formats[p], ParticleFrame.particle_base_formats[norm]), savefig_prefix='Univariate Bayesian Approach: ')

if args.run_univariate_bayes_outliers:
    hold = args.hold
    whis = args.whis
    norm = args.norm

    plt.figure()
    if whis:
        plt.boxplot(data[norm][hold], whis=whis, sym='+')
    else:
        plt.boxplot(data[norm][hold], whis='range', sym='+')
    plt.yscale('log')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel(ParticleFrame.variable_formats[hold] + ' (' + ParticleFrame.variable_units[hold] + ')')
    data.pyplot_sanitize_show('Outliers Outside of ' + str(whis) + ' IQR on a Logarithmic Scale', savefig_prefix='Univariate Bayesian Approach: ')

if args.run_multivariate_bayes:
    cut = args.cut

    holdings = args.holdings
    mc_best = args.mc_best
    exclusive_cut = args.exclusive_cut
    bar_particles = args.bar_particles
    whis = args.whis
    niterations = args.niterations
    nbins = args.nbins
    norm = args.norm
    particles_of_interest = args.particles_of_interest

    cutting_columns, category_columns, intervals, iteration_priors = data.multivariate_bayes(holdings=holdings, whis=whis, norm=norm, mc_best=mc_best, niterations=niterations, nbins=nbins)

    interval_centers = {}
    interval_widths = {}
    for hold in holdings:
        interval_centers[hold] = {key: np.array([np.mean(value[i:i+2]) for i in range(len(value)-1)]) for key, value in intervals[hold].items()}
        interval_widths[hold] = {key: np.array([value[i] - value[i-1] for i in range(1, len(value))]) / 2. for key, value in intervals[hold].items()}

    for p in particles_of_interest:
        fig = plt.figure()
        plt.imshow(np.array(iteration_priors[norm][p]).reshape(nbins, nbins).T, cmap='viridis')
        plt.grid(b=False, axis='both')
        plt.xlabel(ParticleFrame.variable_formats[holdings[0]] + ' (' + ParticleFrame.variable_units[holdings[0]] + ')')
        plt.xticks(range(nbins), interval_centers[holdings[0]][p])
        fig.autofmt_xdate()
        plt.ylabel(ParticleFrame.variable_formats[holdings[1]] + ' (' + ParticleFrame.variable_units[holdings[1]] + ')')
        plt.yticks(range(nbins), interval_centers[holdings[1]][p])
        plt.colorbar()
        data.pyplot_sanitize_show('%s Spectra Ratios Relative to %s'%(ParticleFrame.particle_base_formats[p], ParticleFrame.particle_base_formats[norm]), savefig_prefix='Multivariate Bayesian Approach: ')

    c = data.add_isMax_column(cutting_columns) if exclusive_cut else cutting_columns
    epsilonPIDs = data.epsilonPID_matrix(cutting_columns=c, cut=cut)
    if exclusive_cut:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for an exclusive Cut'
    else:
        drawing_title = r'Heatmap of $\epsilon_{PID}$ Matrix for a Cut at $%.2f$'%(cut)
    data.plot_epsilonPIDs(epsilonPIDs, title=drawing_title, savefig_prefix='Multivariate Bayesian Approach: ')

    data.plot_stats_by_particle(data.stats(cutting_columns=cutting_columns), particles_of_interest=particles_of_interest, savefig_prefix='Multivariate Bayesian Approach: ')
    data.plot_neyman_pearson(cutting_columns=cutting_columns, title_suffix=' by ' + ' & '.join([ParticleFrame.variable_formats[h] for h in holdings]), particles_of_interest=particles_of_interest, bar_particles=bar_particles, savefig_prefix='Multivariate Bayesian Approach: ')

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
    plt.imshow(correlation_matrix, cmap='viridis', vmin=-1., vmax=1.)
    for (j, i), label in np.ndenumerate(correlation_matrix):
        plt.text(i, j, r'$%.2f$'%(label), ha='center', va='center', fontsize='small', color=str(np.piecewise(label, [label < 0, label >= 0], [1, 0])))
    plt.grid(b=False, axis='both')
    plt.xlabel('Predicted Particle')
    plt.xticks(range(len(holdings)), [ParticleFrame.variable_formats[v] for v in holdings])
    plt.ylabel('True Particle')
    plt.yticks(range(len(holdings)), [ParticleFrame.variable_formats[v] for v in holdings])
    plt.colorbar()
    data.pyplot_sanitize_show('Heatmap of Correlation Matrix of ROOT Variables', savefig_prefix='Multivariate Bayesian Approach: ')

    selection = np.ones(particle_data.shape[0], dtype=bool)
    if whis:
        for hold in holdings:
            q75, q25 = np.percentile(particle_data[hold], [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (iqr * whis)
            upper_bound = q75 + (iqr * whis)
            selection = selection & (particle_data[hold] > lower_bound) & (particle_data[hold] < upper_bound)

    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    main_ax = plt.subplot(grid[:-1, 1:])
    # NOTE: This is one of the few place where we differentiate between particle and anti-particle
    for v in np.unique(particle_data[selection]['mcPDG'].values):
        particle_data.at[selection & (particle_data[selection]['mcPDG'] == v), truth_color_column] = list(np.unique(particle_data[selection]['mcPDG'].values)).index(v)
    plt.scatter(particle_data[selection][holdings[0]], particle_data[selection][holdings[1]], c=particle_data[selection][truth_color_column], cmap=plt.cm.get_cmap('viridis', np.unique(particle_data[selection]['mcPDG'].values).shape[0]), s=5., alpha=.1)
    plt.setp(main_ax.get_xticklabels(), visible=False)
    plt.setp(main_ax.get_yticklabels(), visible=False)

    plt.subplot(grid[-1, 1:], sharex=main_ax)
    plt.hist(particle_data[selection][holdings[0]], nbins, histtype='step', orientation='vertical')
    plt.gca().invert_yaxis()
    plt.xlabel(ParticleFrame.variable_formats[holdings[0]] + ' (' + ParticleFrame.variable_units[holdings[0]] + ')')
    plt.subplot(grid[:-1, 0], sharey=main_ax)
    plt.hist(particle_data[selection][holdings[1]], nbins, histtype='step', orientation='horizontal')
    plt.gca().invert_xaxis()
    plt.ylabel(ParticleFrame.variable_formats[holdings[1]] + ' (' + ParticleFrame.variable_units[holdings[1]] + ')')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.20, 0.05, 0.6])
    cbar = plt.colorbar(cax=cbar_ax, ticks=range(np.unique(particle_data[selection]['mcPDG'].values).shape[0]))
    cbar.set_alpha(1.)
    cbar.set_ticklabels([ParticleFrame.particle_formats[lib.pdg_to_name_faulty(v)] for v in np.unique(particle_data[selection]['mcPDG'].values)])
    cbar.draw_all()

    data.pyplot_sanitize_show('Multi-axes Histogram of ' + ', '.join(format(hold) for hold in holdings), savefig_prefix='Multivariate Bayesian Approach: ', suptitle=True, format='png')


plt.show()
