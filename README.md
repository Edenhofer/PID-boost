# Playground repository for boosted particle identification

## Inspiration

The main inspiration for the thesis heavily relies on the research performed at the ALICE Collaboration ([Particle identification in ALICE: a Bayesian approach](https://arxiv.org/abs/1602.01392), [Bayesian Approach for Combined Particle Identification in Alice Experiment at LHC](https://cds.cern.ch/record/865616/files/p423.pdf)).

## Outline of the [thesis](doc/thesis/Optimization of Particle Identification, 29-06-18.pdf)

1. Abstract
* Introduction: Standard Model, flavour physics, ultra-rare decays, CP-Violation
* Belle 2
  * Experiment: About the experiment itself
  * Detector system
  * Interaction with matter: particle identification -- from the initial event to a final PID
* Statistics for particle analysis
  * Classification functions: FPR, TPR, PPV, ...
  * Receiver operating characteristic
  * Identification efficiencies: confusion matrix
  * Likelihood: likelihood ratio, Neyman-Pearson
  * Neural network
* Application
  * Data sample
  * Particle identification variables: inventory
    * Legacy PID
    * Global PID
    * Goodness of the global PID: Default approach for upcoming releases, Neyman-Pearson
  * Bayesian Approach
    * Simple Bayes
    * Univariate Bayes
    * Multivariate Bayes
    * Comparison
    * Summary and outlook
  * Neural network
    * Design: dense layers, drop-out, classification, activation, optimizers
    * Performance
    * Comparison
    * Summary and outlook
* Conclusion
1. Computer code

## Roadmap

* ~~Calculate a sensible ROC figure~~
* ~~Calculate and visualize the epsilon_PID matrix~~
* ~~Visualize precision and recall~~
* ~~Evaluate various "pid*" variables and reconstruct, e.g. the 'pionID' (see `basf2 variables.py` section 'PID', e.g. variable 'pidLogLikelihoodValueExpert')~~
* ~~Construct a naive alternative to the particle ID variable using Bayes and e.g. a flat prior~~
* ~~Compare the naive approach with the particle ID variable value~~
* ~~Allow for variable dependant priors for the bayesian approach, e.g. make the priors depend on the transverse momentum~~
* ~~Compare the dependable priors bayesian approach with the simple bayesian approach and the particle ID one~~
* ~~Allow for multivariate a prior probabilities on two variables~~
* ~~Compare the multi-dependant priors~~
* ~~Introduce an alternative selection methods, i.e. maximum probability~~
* ~~Use different sets of data to diversify results~~
* ~~Add a simple Machine Learning approach using the likelihoods as input and the prediction as output~~
* Design a functional Neural Network classifying particles
  * ~~Utilize automatically selected ROOT variables as input, e.g. depending on their contribution to a PCA~~
  * ~~Use a multitude of variables and let the network decide~~
  * ~~Visually compare the approaches~~
  * ~~Sample the data in a way which makes all particles equally likely~~
  * ~~Pre-train neural network on e.g. data sets containing all particles in the same abundance~~
  * ~~Tweak pre-trained model for decay specific particle abundances~~
* ~~Plot efficiencies over background-ratio for PID, Bayes, etc.~~
* ~~Visualize the difference in assumed particle abundances for various approaches~~
* ~~Abundance comparisons differentiating by correctly and faulty reconstructed ones, e.g. in a stacked histogram~~
* Efficiency and contamination over transverse momentum plot
* Output efficiency tables for various approaches, including e.g. the Area Under Curve for PPV and ROC

## Physics performance meeting

* ~~Confusion matrices of the identification process~~
* ~~`pidProbability` maybe is not set for certain detectors above a specific threshold and therefore screws the result~~
* ~~Kink in proton ID could be caused by the addition of further detectors~~
* ~~Sample `pidProbability` by `pt` and `Theta` bins and then perform the test based on the Neyman-Pearson lemma~~
  * ~~Visualize plot for each hold~~
  * ~~Use a density based approach for visualization of both~~
* ~~Idea do manifest that we are using bins: xerr bar in neyman-pearson~~
* ~~Look up where the detector is and whether it explains the effect seen in the Multi-Axis histogram~~ &rarr; CDC and TOP is reached at `0.28 < pt < 0.52`; Error in CDC detector (by Thomas); ~~End of coverage of CDC at `-0.5` `cosTheta` (my objection)~~
* ~~Retrieve the proton's point of origin and apply strict cuts (mcTruth of vertex)~~ &rarr; There are no proton tracks with `isPrimarySignal`
* ~~Share confusion matrix code~~
* ~~Investigate alternative detector calibrations~~
  * ~~Regenerate charged MC9 data using an altered detector configuration~~
  * ~~Rerun the analysis on the generated data~~
  * ~~Validate the new graphs by rerunning the analysis on generated data with the old calibration~~

### Bayes, lib and beyond

* ~~Add shell completion using bash stuff~~
* ~~Add same completion for bash~~
* ~~Box plot outliers~~
* ~~Strip outliers and plot rates~~
* ~~Investigate why epsilon_PID does not sum to one~~ &rarr; Cut is not exclusive
* ~~Select by theta~~
* ~~Why does the epsilon_PID contain values for the deuterons~~
* ~~Try out seaborn for plotting with custom themes to improve style~~
* ~~Make `stats` honor particles_of_interest~~
* ~~Fancier argparse list handling~~
* ~~Do not choose arbitrary cuts for the diff of epsilonPIDs but instead e.g. try to have both categorize the same number of values as true?~~ --> Better: maximum probability cut
* ~~Rename chunked to univariate or multivariate~~
* ~~Plot particle abundance as histogram~~
* ~~Investigate the possibility of results being screwed by NaN values in priors~~ &rarr; Guess they are okay; If the value is nan then there is no data available to compare the priors to
* ~~Fixup & Improve Bayes~~
  * ~~Find alternatives to and strip data selection by whiskers~~ &rarr; Disable whiskers by default and drop un-physical values prior to precessing anything
  * ~~Look into stats ratios, as they are not in compliance with the given rates~~
  * ~~Is the PID variable simply defined for more rows; I.e. is the denominator including rows with NaN values~~ &rarr; Yes
  * ~~Check that flat_bayes is actually identical to the ROOT variable `pidProbabilityExpert`~~ &rarr; Both are more or less equal; Considering the numerical precision the result is probably fine
  * ~~Investigate particles pairs and Anti-Particles treatment~~
* ~~Keep the window size for stat plots fixed between 0 and 1 for the x- as well as the y-axis~~
* ~~Share the colorbar between plots in diff_epsilon instead of just reusing the last~~
* ~~Diff stats only for range in between minimal and maximal values and no interpolation beyond the data~~
* ~~Fix path joining by not relying on '/' being the default delimiter~~
* ~~Remove title from figure when saving~~
* ~~Plot Neyman-Pearson for different cutting columns, e.g. the Bayesian columns~~ - Bear in mind that it is not straight-forward to compare those graphs
* ~~Plot Neyman-Pearson for each particle and anti-particle separately~~
* ~~Use pandas' query attribute instead of manual selection~~ &rarr; Utilized where appropriate
* ~~Put the font of the label of the epsilonPID cells on a grey-scale depending on their background~~
* ~~Cut data to ignore `mcPDG = 0` (not our beer to remove misidentified particles), `z0 < 5cm`, `d0 < 2cm` and `pt < 50 MeV/c`~~
* ~~Plot number of hits for `mcPDG = 0` for various detectors (ask Jakob) as motivation for excluding those tracks~~
* ~~Make `bar_particles` configurable but keep the non-bar part of it even when 'bared'~~
* ~~Calculate the overall accuracy of an approach (especially useful for comparisons with a neural network)~~
* Save the calculated columns from event_metrics to a pickle file & Check for columns existence prior to calculating the values anew (configurable via '--force')
* ? Utilize pandas' multi-indexing instead of nesting the DataFrames into a dictionary

### Neural network

* ~~Start by inputting logLikelihoods and maybe add a sum function as node~~
* ~~Allow the adaption of the `epochs`, `training_fraction`, etc. variable~~
* ~~Include the NN model in the ParticleFrame and save it into the pickle~~ &rarr; Bad idea as it will unnecessarily clutter the file with unrelated data
* ~~Add model-saving checkpoints as callback~~

## Useful snippets and code examples

### Source code snippets

* Selecting pidProbabilities for the `SVD` detector

```python
d = 'svd'
c = {k: 'pidProbabilityExpert__bo' + basf2_Code(k) + '__cm__sp' + d + '__bc' for k in particles}
title_suffixes += [' via SVD PID']
```

* Library

```python
def __init__(self, particles=None, particle_base_formats=None, detectors=None, pseudo_detectors=None, detector_weights=None, physical_boundaries=None):
    """Base definitions of stable particles and detector components to be included in calculations."""

    self.particles = ['K+', 'pi+', 'e+', 'mu+', 'p+', 'deuteron'] if particles is None else particles
    self.detectors = ['pxd', 'svd', 'cdc', 'top', 'arich', 'ecl', 'klm'] if detectors is None else detectors
    self.pseudo_detectors = ['all', 'default'] if pseudo_detectors is None else pseudo_detectors
    # Use the detector weights to exclude certain detectors, e.g. for debugging purposes
    # Bear in mind that if all likelihoods are calculated correctly this should never improve the result
    self.detector_weights = {d: 1. for d in ['pxd', 'svd', 'cdc', 'top', 'arich', 'ecl', 'klm', 'all', 'default']} if detector_weights is None else detector_weights
    # Dictionary of variables and their boundaries for possible values they might yield
    self.physical_boundaries = {'pt': (0, 5.5), 'cosTheta': (-1, 1)} if physical_boundaries is None else physical_boundaries
```

* Search for differences in pidProbabilities for particles and ant-particles (Bear in mind `np.nan == np.nan` is `False`) &rarr; There are none!

```python
for p, anti_p in zip(particles, particles_bar):
     print(pf[p].dropna()[pf[p]['pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc'] != pf[p]['pidProbabilityExpert__bo' + lib.basf2_Code(anti_p) + '__cm__sp' + d + '__bc']][['pidProbabilityExpert__bo' + lib.basf2_Code(p) + '__cm__sp' + d + '__bc', 'pidProbabilityExpert__bo' + lib.basf2_Code(anti_p) + '__cm__sp' + d + '__bc']])
```

* Motivation for excluding `mcPDG = 0`

```python
nbins = 10; ParticleFrame = lib.ParticleFrame
# NOTE: This will taint the class
ParticleFrame.physical_boundaries = {'0.05 < pt < 5.29', 'z0 < 5', 'd0 < 2'}

data = ParticleFrame()
data.read_root('./charged.root/task01')

for p, detector_hits in itertools.product(ParticleFrame.particles, ['nCDCHits', 'nPXDHits', 'nSVDHits', 'nVXDHits']):
  plt.figure()
  plt.hist(data[p].query('mcPDG == 0')[detector_hits], nbins, normed=True, histtype='step', label='False (# %d)'%(data[p].query('mcPDG == 0')[detector_hits].shape[0]))
  plt.hist(data[p].query('mcPDG != 0')[detector_hits], nbins, normed=True, histtype='step', label='True (# %d)'%(data[p].query('mcPDG != 0')[detector_hits].shape[0]))
  plt.legend()
  title = detector_hits + ' for ' + ParticleFrame.particle_base_formats[p]
  sanitized_title = re.sub('[\\\\$_^{}]', '', 'General Purpose Statistics: ' + title + '')
  plt.savefig(os.path.join('./doc/res/charged 01/', sanitized_title + '.' + 'pdf'), bbox_inches='tight', format='pdf')
  plt.title(title)
  plt.show(block=False)

plt.show()
```

* Receiver Operating Characteristic curve

```python
x = np.linspace(0., 1., 1000)
title = 'Sample Receiver Operating Characteristic (ROC) curve'

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x, x, linestyle='--', label='Guessing')
plt.plot(x, x**4, label='Worse')
plt.plot(x, x**(1/4), label='Better')
plt.legend()
plt.savefig(os.path.join('./doc/res/', title + '.' + 'pdf'), bbox_inches='tight', format='pdf')
plt.title(title)
plt.show()
```

* Neyman-Pearson visualization

```python
x = np.linspace(0.05, 0.95, 10)
y = np.exp(x)
title = 'Neyman-Pearson Visualization'

plt.xlabel(r'$\mathcal{LR}(\pi)$')
plt.ylabel(r'$\frac{\#\pi}{\sum_{a}{\# A}}$')
plt.errorbar(x, y/y.max(), xerr=0.05, capsize=0., elinewidth=1, marker='o', markersize=4, markeredgewidth=1, markerfacecolor='None', linestyle='--', linewidth=0.1)
plt.savefig(os.path.join('./doc/res/', title + '.' + 'pdf'), bbox_inches='tight', format='pdf')
plt.title(title)
plt.show()
```

## Job scheduling

* Overnight NTuples creation

```bash
for dir in '.' 'y2s.root' 'y2s.root/noise/task01' 'charged.root/task01' 'charged.root/task02' 'charged.root/task03' 'charged.root/task04' 'mixed.root/task01' 'mixed.root/task02' 'mixed.root/task03' 'mixed.root/task04'; do
  (echo ${dir}; cd ${dir} && ls -AlFh *mdst*.root && basf2 ~/Work/PID-boost/event_preprocessing.py -p4 -i "*mdst*.root")
done; \
```

* Overnight event-metrics job scheduling

```bash
PARAMS=('--info' '--mc-best' '--exclusive-cut' '--nbins=10' '--ncuts=40' '--holdings' 'pt' 'cosTheta' '--norm' 'pi+');
tasks=('--multivariate-bayes-motivation' '--multivariate-bayes' '--univariate-bayes-outliers' '--niterations=0 --univariate-bayes-priors' '--hold cosTheta --niterations=0 --univariate-bayes-priors' '--univariate-bayes-priors' '--univariate-bayes' '--bayes' '--pid' '--stats' '--bar-particles --pidProbability' '--bar-particles --norm p+ --pidProbability vertex' '--bar-particles --norm p+ --pidProbability primary' '--pidProbability-motivation' '--diff pid flat_bayes' '--diff pid simple_bayes' '--diff pid univariate_bayes' '--diff pid multivariate_bayes' '--diff pidProbability flat_bayes' '--diff pidProbability simple_bayes' '--diff pidProbability univariate_bayes' '--diff pidProbability multivariate_bayes' '--diff flat_bayes simple_bayes' '--diff flat_bayes univariate_bayes' '--diff flat_bayes multivariate_bayes' '--diff simple_bayes univariate_bayes' '--diff simple_bayes multivariate_bayes' '--diff univariate_bayes multivariate_bayes' '--diff univariate_bayes_by_cosTheta multivariate_bayes');
for i in "${tasks[@]}"; do >&2 echo "$(echo ${i})" && event_metrics.py --non-interactive -i . -o ./doc/res/"sample" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i y2s.root -o ./doc/res/"y2s" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do >&2 echo "$(echo ${i})" && event_metrics.py --non-interactive -i charged.root/task01 -o ./doc/res/"charged 01" ${PARAMS[@]} $(echo ${i}); done; \

for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i mixed.root/task01 -o ./doc/res/"mixed 01" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i charged.root/task02 -o ./doc/res/"charged 02" ${PARAMS[@]} $(echo ${i}); done; \

for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i mixed.root/task02 -o ./doc/res/"mixed 02" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i charged.root/task03 -o ./doc/res/"charged 03" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i mixed.root/task03 -o ./doc/res/"mixed 03" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do event_metrics.py --non-interactive -i charged.root/task04 -o ./doc/res/"charged 04" ${PARAMS[@]} $(echo ${i}); done; \

tasks=('--pid' '--stats' '--bar-particles --pidProbability' '--bar-particles --norm p+ --pidProbability vertex' '--bar-particles --norm p+ --pidProbability primary' '--pidProbability-motivation')
for i in "${tasks[@]}"; do >&2 echo "$(echo ${i})" && event_metrics.py --non-interactive -i unset.root -o ./doc/res/"unset" ${PARAMS[@]} $(echo ${i}); done; \
for i in "${tasks[@]}"; do >&2 echo "$(echo ${i})" && event_metrics.py --non-interactive -i set.root -o ./doc/res/"set" ${PARAMS[@]} $(echo ${i}); done; \
```

* Overnight neural network training

```bash
NN_PARAMS=('--info' '--add-dropout' '--batch-size=256' '--training-fraction=0.80' '--ncomponents=70' '--epochs=25' '--gpu=0' '--log-dir=/dev/null');
nn_tasks=('--sampling-method fair --pidProbability' '--sampling-method biased --pidProbability' '--sampling-method fair --all' '--sampling-method biased --all');
nn_optimizer_methods=('rmsprop' 'sgd' 'adagrad' 'adadelta' 'adam' 'adamax' 'nadam');

for i in "${nn_tasks[@]}"; do
  for n in "${nn_optimizer_methods[@]}"; do
    event_nn.py --non-interactive -i charged.root/task01 -o /srv/data/$(id -un)/res/"charged 01" ${NN_PARAMS[@]} --optimizer $(echo ${n}) $(echo ${i});
  done;
done; \
for i in "${nn_tasks[@]}"; do
  for n in "${nn_optimizer_methods[@]}"; do
    event_nn.py --non-interactive -i mixed.root/task01 -o /srv/data/$(id -un)/res/"mixed 01" ${NN_PARAMS[@]} --optimizer $(echo ${n})  $(echo ${i});
  done;
done; \

NN_PARAMS=('--info' '--add-dropout' '--batch-size=256' '--training-fraction=0.80' '--ncomponents=90' '--epochs=25' '--gpu=0' '--log-dir=/dev/null');
nn_tasks=('--sampling-method fair --pca' '--sampling-method biased --pca');
nn_optimizer_methods=('adadelta' 'adamax');

for i in "${nn_tasks[@]}"; do
  for n in "${nn_optimizer_methods[@]}"; do
    event_nn.py --non-interactive -i charged.root/task01 -o /srv/data/$(id -un)/res/"charged 01" ${NN_PARAMS[@]} --optimizer $(echo ${n}) $(echo ${i});
  done;
done; \
```

* Neural network applying models and calculating metrics

```bash
NN_PARAMS=('--info' '--add-dropout' '--batch-size=256' '--training-fraction=0.80' '--ncomponents=70' '--epochs=25' '--gpu=0' '--log-dir=/dev/null');
event_nn.py --non-interactive -i charged.root/task04/ -o res/"charged 04" ${NN_PARAMS[@]} --apply --input-module "model_checkpoint_pca_ncomponents90_biased_nAdditionalLayers1_Optimizeradamax_LearningRateNone_nEpochs25_BatchSize256.h5" --input-pca "pca_ncomponents90.pkl"

PARAMS=('--info' '--mc-best' '--exclusive-cut' '--nbins=10' '--ncuts=40' '--holdings' 'pt' 'cosTheta' '--norm' 'pi+');
tasks=('--diff pid nn' '--diff pidProbability nn' '--diff flat_bayes nn' '--diff univariate_bayes nn' '--diff multivariate_bayes nn')
for i in "${tasks[@]}"; do
  >&2 echo "$(echo ${i})" \
  && event_metrics.py --non-interactive -o doc/res/"charged 04" ${PARAMS[@]} --input-pickle "res/charged 04/ParticleFrame_applypca_ncomponents90_biased_nAdditionalLayers1_Optimizeradamax_LearningRateNone_nEpochs25_BatchSize256.pkl" $(echo ${i});
done; \
```

* Neural network model visualization

```bash
for i in res/"sample"/history_*.pkl; do event_model.py --non-interactive -o doc/res/"sample" --history-file "${i}"; done; \
for i in res/"y2s"/history_*.pkl; do event_model.py --non-interactive -o doc/res/"y2s" --history-file "${i}"; done; \
for i in res/"charged 01"/history_*.pkl; do event_model.py --non-interactive -o doc/res/"charged 01" --history-file "${i}"; done; \
for i in res/"mixed 01"/history_*.pkl; do event_model.py --non-interactive -o doc/res/"mixed 01" --history-file "${i}"; done; \
```

* MC9 charged event generation

```bash
source ~/setup_belle2.sh release-01-00-03
(unset BELLE2_CONDB_GLOBALTAG; for n in {10..20}; do &>unset_$n.log basf2 generate_charged.py -n 10000 -o unset_$n.mdst.root &; done)
(export BELLE2_CONDB_GLOBALTAG="Calibration_Offline_Development"; for n in {10..20}; do &>set_$n.log basf2 generate_charged.py -n 10000 -o set_$n.mdst.root &; done)
```
