# Simulated Annealing

This is an implementation of simulated annealing adapted to minimize chi2 of some model fitting data.

the function `param_locked_SA` iteratively locks the fitting parameters and optimizes the rest to characterize the dependence of each parameter on chi2, and thus get confidence intervals (i.e. Avni et al 1976).


