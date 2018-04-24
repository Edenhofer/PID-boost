# Playground repository for boosted Particle IDentification

## Inspiration

The main inspiration for the actual program code heavily relies on the [b2-starterkit](https://stash.desy.de/scm/~ritter/b2-starterkit.git) tutorial.

## Roadmap

* ~~Calculate a sensible ROC figure~~
* ~~Calculate and visualize the epsilon_PID matrix~~
* ~~Visualize precision and recall~~
* ~~Evaluate various "pid*" variables and reconstruct, e.g. the 'pionID' (see `basf2 variables.py` section 'PID', e.g. variable 'pidLogLikelihoodValueExpert')~~
* Construct a naive alternative to the particle ID variable using Bayes and e.g. a flat prior
* Compare the naive approach with the particle ID variable value
