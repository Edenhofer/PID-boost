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
* ~~Use different sets of data to diversify results~~
* ~~Add a simple Machine Learning approach using the likelihoods as input and the prediction as output~~
* Design a functional Neural Network classifying particles
  * ~~Utilize automatically selected ROOT variables as input, e.g. depending on their contribution to a PCA~~
  * Use a multitude of variables and let the network decide
  * ~~Visually compare the approaches~~
  * ~~Sample the data in a way which makes all particles equally likely~~
  * ~~Pre-train neural network on e.g. data sets containing all particles in the same abundance~~
  * Tweak pre-trained model for decay specific particle abundances
* Plot efficiencies over background-ratio for PID, Bayes, etc. and plot a dot for the NN
* ~~Visualize the difference in assumed particle abundances for various approaches~~
* Efficiency and contamination over transverse momentum plot
* Output efficiency tables for various approaches, including e.g. the Area Under Curve for PPV and ROC
