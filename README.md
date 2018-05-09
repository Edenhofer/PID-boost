# Playground repository for boosted Particle Identification

## Inspiration

The main inspiration for the actual program code heavily relies on the [b2-starterkit](https://stash.desy.de/scm/~ritter/b2-starterkit.git) tutorial.

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
* Add a simple Machine Learning approach using the likelihoods as input and the prediction as output
