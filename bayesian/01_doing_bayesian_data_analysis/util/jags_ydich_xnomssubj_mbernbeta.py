import numpy as np
import pandas as pd
import pymc3 as pm
from math import ceil
import matplotlib.pyplot as plt
from dbda2e_utilities import summarize_post, plot_post

def gen_mcmc(data, num_saved_steps = 50000):
    # This function expects the data to be a Pandas dataframe with one column named y
    # being a Pandas series of integer 0, 1 values and one column named s being a 
    # Pandas series of subject identifiers

    # THE DATA
    y = data.y.astype(int).values
    # Convert strings to consecutive integer levels
    subjects = data.s.unique().tolist()
    s = data.s.apply(lambda x: subjects.index(x))
    # Do some checking that data makes sense
    if np.any(np.logical_and(y != 0, y != 1)):
        raise ValueError('All y values must be 0 or 1')
    n_total = len(y)
    n_subj = len(subjects)

    # THE MODEL
    theta = []
    y_obs = []
    parameters = [] # The parameters to be monitored
    with pm.Model() as model:
        for i in range(len(subjects)):
            # N.B. 2,2 prior; change as appropriate
            # No need to use the 'start' parameter in pm.Beta since the auto selected 
            # MCMC algorithm already initialises the chain's starting point 
            # )In the original R code the starting point was the MLE + some random noise)
            theta.append(pm.Beta('theta_' + str(i + 1), alpha = 2, beta = 2))
            parameters.append('theta_' + str(i + 1))
            y_obs.append(pm.Bernoulli('y_obs_' + str(i + 1), p = theta[i], observed = y[s == i]))
        # RUN THE CHAINS - some of the parameters in the original R code do not translate
        # well to the PyMC3's MCMC algorithm (adapt_stes, burn_in_Steps)
        n_chains = 4        # n_chains should be 2 or more for diagnostics
        thin_steps = 1
        n_iter = ceil((num_saved_steps * thin_steps) / float(n_chains))
        trace = pm.sample(chains = n_chains, draws = n_iter)

    return trace

def smry_mcmc(trace, comp_val = 0.5, rope = None, comp_val_diff = 0, rope_diff = None):
    parameter_names = [v for v in trace.varnames if 'logodds' not in v]
    mcmc_mat = pd.DataFrame()
    for p in parameter_names:
        mcmc_mat[p] = trace[p]
    summary_info = {}
    for p in parameter_names:
        summary_info[p] = summarize_post(mcmc_mat[p].values.tolist(), 
                                           comp_val = comp_val,
                                           rope = rope)
    for t1_idx in range(len(parameter_names)):
        for t2_idx in range(t1_idx + 1, len(parameter_names)):
            par_name1 = parameter_names[t1_idx]
            par_name2 = parameter_names[t2_idx]
            summary_info[par_name1 + \
                         ' - ' + \
                         par_name2] = summarize_post((mcmc_mat[par_name1] - \
                                                      mcmc_mat[par_name2]).values \
                                                                          .tolist(),
                                                     comp_val = comp_val_diff,
                                                     rope = rope_diff)

    return(summary_info)

def plot_mcmc(trace, data, comp_val = 0.5, rope = None, comp_val_diff = 0, rope_diff = None, cen_tend = None, cen_tend_diff = None):
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
    n_theta = len(parameter_names)
    fig, ax = plt.subplots(n_theta, n_theta)
    for t1_idx in range(n_theta):
        for t2_idx in range(n_theta):
            par_name1 = 'theta_' + str(t1_idx + 1)
            par_name2 = 'theta_' + str(t2_idx + 1)
            if t1_idx > t2_idx:
                n_to_plot = 700
                pt_idx = np.around(np.linspace(0, len(mcmc_mat) - 1, n_to_plot))
                ax[t1_idx][t2_idx].scatter(mcmc_mat[par_name2][pt_idx], mcmc_mat[par_name1][pt_idx], alpha = 0.5)
                ax[t1_idx][t2_idx].set_xlabel(par_name2)
                ax[t1_idx][t2_idx].set_ylabel(par_name1)
            elif t1_idx == t2_idx:
                post_info = plot_post(mcmc_mat[par_name1], 
                                      ax = ax[t1_idx][t2_idx], 
                                      comp_val = comp_val, 
                                      rope = rope, 
                                      xlab = par_name1, 
                                      main = '',
                                      cen_tend = cen_tend)
                include_rows = (s == t1_idx) # Identify rows of this subject in the data
                data_propor = np.sum(y[include_rows]) / np.sum(include_rows)
                ax[t1_idx][t2_idx].plot(data_propor, 0, '+', c = 'red', markersize = 10)
            elif t1_idx < t2_idx:
                post_info = plot_post(mcmc_mat[par_name1] - mcmc_mat[par_name2],
                                      ax = ax[t1_idx][t2_idx],
                                      comp_val = comp_val_diff,
                                      rope = rope_diff,
                                      xlab = par_name1 + ' - ' + par_name2,
                                      main = '',
                                      cen_tend = cen_tend_diff)
                include_rows1 = (s == t1_idx) # Identify rows of this subject in the data
                data_propor1 = np.sum(y[include_rows1]) / np.sum(include_rows1)
                include_rows2 = (s == t2_idx) # Identify rows of this subject in the data
                data_propor2 = np.sum(y[include_rows2]) / np.sum(include_rows2)
                ax[t1_idx][t2_idx].plot(data_propor1 - data_propor2, 0, '+', c = 'red', markersize = 10)

    fig.set_figwidth(12)
    fig.set_figheight(6)
    plt.tight_layout()