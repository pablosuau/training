import numpy as np
import pymc3 as pm
from math import ceil
from dbda2e_utilities import diag_mcmc, plot_post

# THE DATA
n = 30
z = ceil(0.55 * n)
y = np.hstack((np.repeat(0, n - z), np.repeat(1, z))).tolist()

def _define_deterministic(name, m, priors):
  return pm.Deterministic(name, 
                          (1 - m) * priors[0] + \
                          m * priors[1])

# THE MODEL
with pm.Model() as model:
  m_prior_prob = [0.5, 0.5] 
  m = pm.Categorical('m', p = m_prior_prob)
  kappa1_prior = np.array([20, 50]) # True prior and pseudo prior values
  kappa1 = _define_deterministic('kappa1', m, kappa1_prior)
  omega1_prior = np.array([0.1, 0.4]) # True prior and pseudo prior values
  omega1 = _define_deterministic('omega1', m, omega1_prior)
  kappa2_prior = np.array([50, 20]) # Pseudo prior and true prior values
  kappa2 = _define_deterministic('kappa2', m, kappa2_prior)
  omega2_prior = np.array([0.7, 0.9]) # Pseudo prior and true prior values
  omega2 = _define_deterministic('omega2', m, omega2_prior)
  theta1 = pm.Beta('theta1',
                   alpha = omega1 * (kappa1 - 2) + 1,
                   beta = (1 - omega1) * (kappa1 - 2) + 1)
  theta2 = pm.Beta('theta2',
                   alpha = omega2 * (kappa2 - 2) + 1,
                   beta = (1 - omega2) * (kappa2 - 2) + 1)
  theta = pm.Deterministic('theta', 
                           (1 - m) * theta1 + \
                           m * theta2)
  for i in range(n):
    _ = pm.Bernoulli('z_obs_' + str(i), p = theta, observed = y[i])

  # RUN THE CHAINS
  # Test
  n_chains = 4
  num_saved_steps = 10000
  thin_steps = 1
  # RUN THE CHAINS - some of the parameters in the original R code do not translate
  # well to the PyMC3's MCMC algorithm (thin_Steps, burn_in_steps)
  n_per_chain = ceil((num_saved_steps * thin_steps) / float(n_chains))
  trace = pm.sample(chains = n_chains, draws = n_per_chain)

  parameter_names = ['theta1', 'theta2', 'm']
  for par_name in parameter_names:
    diag_mcmc(trace, par_name = par_name)

  # EXAMINE THE RESULTS

  # Convert trace samples to matrix objects for easier handling
  m = trace['m']
  theta1 = trace['theta1']
  theta2 = trace['theta2']

  # Compute the proportion of m at each index value
  pm1 = np.sum(m == 0) / len(m)
  pm2 = 1 - pm1
 
  # Extract theta values for each model index
  theta1m1 = theta1[m == 0]
  theta1m2 = theta1[m == 1]
  theta2m1 = theta2[m == 0]
  theta2m2 = theta2[m == 1]

  # Plot histograms of sampled theta values for each model
  plot_post(m, 
            cen_tend = 'mean', 
            xlab = 'm', 
            main = 'model index, p(m=1|D) = %.3f, p(m=2|D) = %.3f' % (round(pm1, 3), round(pm2, 3)))
  plot_post(theta1m1, 
            main = 'theta1 when m=1 (using true prior)',
            xlab = 'theta1',
            xlim = [0, 1])
  plot_post(theta2m1, 
            main = 'theta2 when m=1; pseudo-prior',
            xlab = 'theta2',
            xlim = [0, 1])
  plot_post(theta1m2, 
            main = 'theta1 when m=2 ; pseudo-prior',
            xlab = 'theta1',
            xlim = [0, 1])
  plot_post(theta2m2, 
            main = 'theta2 when m=1 (using true prior)',
            xlab = 'theta2',
            xlim = [0, 1])