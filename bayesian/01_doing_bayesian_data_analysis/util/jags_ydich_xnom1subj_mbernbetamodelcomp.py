import numpy as np
import pymc3 as pm
from math import ceil
from dbda2e_utilities import diag_mcmc, plot_post

# THE DATA
n = 9
z = 6
y = np.hstack((np.repeat(0, n - z), np.repeat(1, z))).tolist()

# THE MODEL
with pm.Model() as model:
  m_prior_prob = [0.5, 0.5] 
  m = pm.Categorical('m', p = m_prior_prob)
  kappa = 12
  omega = np.array([0.25, 0.75])
  thetas = pm.Beta('thetas',
                   alpha = omega * (kappa - 2) + 1,
                   beta = (1 - omega) * (kappa - 2) + 1,
                   shape = 2)
  theta = pm.Deterministic('theta', 
                           (1 - m) * thetas[0] + \
                           m * thetas[1])
  for i in range(n):
    _ = pm.Bernoulli('z_obs_' + str(i), p = theta, observed = y[i])

  # RUN THE CHAINS
  # Test
  n_chains = 4
  num_saved_steps = 50000
  thin_steps = 1
  # RUN THE CHAINS - some of the parameters in the original R code do not translate
  # well to the PyMC3's MCMC algorithm (thin_Steps, burn_in_steps)
  n_per_chain = ceil((num_saved_steps * thin_steps) / float(n_chains))
  trace = pm.sample(chains = n_chains, draws = n_per_chain)

  parameter_names = ['theta', 'm']
  for par_name in parameter_names:
    diag_mcmc(trace, par_name = par_name)

  # EXAMINE THE RESULTS

  # Convert trace samples to matrix objects for easier handling
  m = trace['m']
  theta = trace['theta']

  # Compute the proportion of m at each index value
  pm1 = np.sum(m == 0) / len(m)
  pm2 = 1 - pm1
 
  # Extract theta values for each model index
  thetam1 = theta[m == 0]
  thetam2 = theta[m == 1]

  # Plot histograms of sampled theta values for each model
  plot_post(m, cen_tend = 'mean', xlab = 'm', main = 'model index')
  plot_post(thetam1, 
            main = 'theta when m=1 ; p(m=1|D) = %.3f' % round(pm1, 3),
            xlab = 'theta',
            xlim = [0, 1])
  plot_post(thetam2, 
            main = 'theta when m=2 ; p(m=2|D) = %.3f' % round(pm2, 3),
            xlab = 'theta',
            xlim = [0, 1])
