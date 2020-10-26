#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import npkw


# Two Bernoulli distributions
p1 = np.array([0.2, 0.8])
p2 = np.array([0.8, 0.2])

# Set both cost coefficients to 20
horizon = 21
error_cost = 20

# Initialize test, call without the 'verbose' parameter or set 'verbose=False' to supress output
npkwt = npkw.Test(horizon, p1, p2, error_cost, verbose=True)

# Calculate all required functions
npkwt.setup()

# The sample is represented by a tuple that counts the occurances of each 
# outcome. For example, x = (3, 2) means '0' has been observed three times and
# '1' has been observed two times.
x = (1, 0)

# The functions rho and d can be evaluated for a given sample and a given value
# of z0, for example
z0 = 0.5
rho_val = npkwt.rho(z0, x)
d_val = npkwt.d(z0, x)

# The least favorable distributions can be accessed via .q
lfds = npkwt.q(z0, x)

# The vector of probabilities to continue the test can be accessed via .psi. 
# Each entry of this vector denotes the probability of continuing the test 
# after the corresponding next observation; here, 0 or 1.
psi_val = npkwt.psi(z0, x)

# The expected remainig sample size can be accessed via .gamma
gamma_val = npkwt.gamma(z0, x)


# For convinience, rho can be obtained as functoins of z0 for a given sample. 
rho = npkwt.get_rho(x)

# Evaluate rho and its derivative via .f and .df
rho_val = rho.f(z0)
drho_val = rho.df(z0)


# rho.f and rho.df only take scalar arguments, for plotting it is convinient
# to vecotrize them
rho_f = np.vectorize(rho.f)
rho_df = np.vectorize(rho.df)
z0 = np.linspace(0, 1, 1001)

plt.plot(z0, rho_f(z0), z0, rho_df(z0))


# The NP-KWT can be simulated under the LFDs or under a user-defined pmf p. 
# The functoin returns a decision for H1 (1) or H2 (2) 
H_q  = npkwt.simulate(verbose=True)         # under LFDs
H_p1 = npkwt.simulate(p=p1, verbose=True)   # under H1
H_p2 = npkwt.simulate(p=p2, verbose=True)   # under H2



