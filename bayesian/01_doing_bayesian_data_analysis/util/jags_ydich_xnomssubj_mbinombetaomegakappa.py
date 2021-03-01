import numpy as np
import pandas as pd
import pymc3 as pm
from math import ceil
import matplotlib.pyplot as plt
from dbda2e_utilities import summarize_post, plot_post

def gen_mcmc(data, s_name = 's', y_name = 'y', num_saved_steps = 50000, thin_steps = 1, n_chains = 4):
    # This function expects the data to be a Pandas dataframe with one column named y
    # being a Pandas series of integer 0, 1 values and one column named s being a 
    # Pandas series of subject identifiers

    # THE DATA
    # Convert strings to consecutive integer levels
    data[y_name] = data[y_name].astype(int)
    # Do some checking that data makes sense
    if np.any(np.logical_and(data[y_name].values != 0, data[y_name].values != 1)):
        raise ValueError('All y values must be 0 or 1')
    z = data[data[y_name] == 1].groupby(s_name).sum().rename(columns = {'y': 'z'})
    n = data.groupby(s_name).count().rename(columns = {'y': 'n'})
    df = n.merge(z, left_index = True, right_index = True)
    subjects = data.s.unique().tolist()

    # THE MODEL
    with pm.Model() as model:
        omega = pm.Beta('omega', 1, 1)
        kappa_minus_two = pm.Gamma('kappa_minus_two', 1.105125, 0.1051249) # mode = 1, sd = 10
        kappa = kappa_minus_two + 2
        
        # The code below is the correct way of dealing with many parameters, instead of using 
        # a for loop. This code is inspired by:
        # https://ericmjl.github.io/bayesian-analysis-recipes/notebooks/hierarchical-baseball/
        thetas = pm.Beta('thetas', 
                         alpha = omega * (kappa - 2) + 1, 
                         beta = (1 - omega) * (kappa - 2) + 1, 
                         shape = len(subjects))
        _ = pm.Binomial('z_obs',  n = df['n'], p = thetas, observed = df['z'])

        # RUN THE CHAINS - some of the parameters in the original R code do not translate
        # well to the PyMC3's MCMC algorithm (adapt_stes, burn_in_Steps)
        n_iter = ceil((num_saved_steps * thin_steps) / float(n_chains))
        trace = pm.sample(chains = n_chains, draws = n_iter)

    for i in range(len(subjects)): 
      trace.add_values({'theta_' + str(i): trace['thetas'][:, i]})
    trace.varnames.remove('thetas')

    return trace

def smry_mcmc(trace, comp_val = 0.5, rope = None, diff_id_vec = None, comp_val_diff = 0, rope_diff = None):
  parameter_names = [v for v in trace.varnames if 'log' not in v]
  mcmc_mat = pd.DataFrame()
  for p in parameter_names:
        mcmc_mat[p] = trace[p]
  thetas = mcmc_mat.columns[mcmc_mat.columns.str.contains('theta')]
  n_theta = len(thetas)
  summary_info = {}
  # overall omega
  summary_info['omega'] = summarize_post(mcmc_mat['omega'].values.tolist(),
                                         comp_val = comp_val,
                                         rope = rope)
  # kappa
  summary_info['kappa_minus_two'] = summarize_post(mcmc_mat['kappa_minus_two'].values.tolist(),
                                         comp_val = None,
                                         rope = None)
  # individual thetas
  for theta in thetas:
      summary_info[theta] = summarize_post(mcmc_mat[theta].values.tolist(),
                                         comp_val = comp_val,
                                         rope = rope)

  # differences of individual theta's
  if diff_id_vec is not None:
      n_idx = len(diff_id_vec)
      for t1_idx in range(n_idx):
          for t2_idx in range(t1_idx + 1, n_idx):
              par_name_1 = thetas[diff_id_vec[t1_idx]]
              par_name_2 = thetas[diff_id_vec[t2_idx]]
              summary_info[par_name_1 + \
                           '-' + \
                           par_name_2] = summarize_post(mcmc_mat[par_name_1] - mcmc_mat[par_name_2],
                                                        comp_val = comp_val_diff,
                                                        rope = rope_diff) 


def plot_mcmc(trace, data, comp_val = 0.5, rope = None, diff_id_vec = None, comp_val_diff = 0.0 , rope_diff = None):
  # N.B.: This function expects the data to be a pandas dataframe,
  # with one component named y being a series of integer 0, 1 values,
  # and one component named s being a series of subject identifiers
  y = data.y.astype(int).values
  # Convert strings to consecutive integer levels
  subjects = data.s.unique().tolist()
  s = data.s.apply(lambda x: subjects.index(x))
  # Now plot the posterior
  parameter_names = [v for v in trace.varnames if 'logodds' not in v]
  mcmc_mat = pd.DataFrame()
  for p in parameter_names:
      mcmc_mat[p] = trace[p]
  chain_length = len(mcmc_mat)
  # Plot omega, kappa
  fig, (ax1, ax2) = plt.subplots(2)
  post_info = plot_post(mcmc_mat['kappa_minus_two'], 
                        ax = ax1, 
                        comp_val = None, 
                        rope = None, 
                        xlab = 'kappa_minus_two', 
                        main = '',
                        xlim = [mcmc_mat['kappa_minus_two'].min(), 
                                mcmc_mat['kappa_minus_two'].quantile(0.99)])
  post_info = plot_post(mcmc_mat['omega'], 
                        ax = ax2, 
                        comp_val = comp_val, 
                        rope = rope, 
                        xlab = 'omega', 
                        main = 'group mode',
                        xlim = [mcmc_mat['omega'].quantile(0.005), 
                                mcmc_mat['omega'].quantile(0.99)])
  # Plot individual theta's and differences
  if diff_id_vec is not None and len(diff_id_vec) > 1:
    n_idx = len(diff_id_vec)
    fig, ax = plt.subplots(n_idx, n_idx)
    for t1_idx in range(n_idx):
      for t2_idx in range(n_idx):
        par_name1 = 'theta_' + str(diff_id_vec[t1_idx])
        par_name2 = 'theta_' + str(diff_id_vec[t2_idx])
        if t1_idx > t2_idx:
          n_to_plot = 700
          pt_idx = np.around(np.linspace(0, chain_length - 1, n_to_plot))
          ax[t1_idx][t2_idx].scatter(mcmc_mat[par_name2][pt_idx], 
                                     mcmc_mat[par_name1][pt_idx], 
                                     alpha = 0.5)
          ax[t1_idx][t2_idx].set_xlabel(par_name2)
          ax[t1_idx][t2_idx].set_ylabel(par_name1)
        elif t1_idx == t2_idx:
          post_info = plot_post(mcmc_mat[par_name1], 
                                ax = ax[t1_idx][t2_idx], 
                                comp_val = comp_val, 
                                rope = rope, 
                                xlab = par_name1, 
                                main = '')
          include_rows = (s == t1_idx) # Identify rows of this subject in the data
          data_propor = np.sum(y[include_rows]) / np.sum(include_rows)
          ax[t1_idx][t2_idx].plot(data_propor, 0, '+', c = 'red', markersize = 10)
        elif t1_idx < t2_idx:
          post_info = plot_post(mcmc_mat[par_name1] - mcmc_mat[par_name2],
                                ax = ax[t1_idx][t2_idx],
                                comp_val = comp_val_diff,
                                rope = rope_diff,
                                xlab = par_name1 + ' - ' + par_name2,
                                main = '')
          include_rows1 = (s == t1_idx) # Identify rows of this subject in the data
          data_propor1 = np.sum(y[include_rows1]) / np.sum(include_rows1)
          include_rows2 = (s == t2_idx) # Identify rows of this subject in the data
          data_propor2 = np.sum(y[include_rows2]) / np.sum(include_rows2)
          ax[t1_idx][t2_idx].plot(data_propor1 - data_propor2, 
                                  0, 
                                  '+', 
                                  c = 'red', 
                                  markersize = 10)

  fig.set_figwidth(12)
  fig.set_figheight(6)
  plt.tight_layout()
