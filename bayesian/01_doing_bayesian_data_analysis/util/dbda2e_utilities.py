import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from scipy.optimize import fmin
from scipy.stats import *
from statsmodels.tsa.stattools import acf
import pymc3 as pm

#------------------------------------------------------------------------------
# Implementation of R functions not in Python
def effective_size(x):
    # Where x is a list
    acf_x = acf(x, fft = False)
    # In python the acf at lag 0 is returned - we have to get rid of it
    acf_x = [f for f in acf_x[1:] if f >= 0.05]
    
    return len(x) / (1 + 2 * np.sum(acf_x))

#------------------------------------------------------------------------------
# Functions for computing limits of HDI's:
def hdi_of_mcmc(sample_vec, cred_mass = 0.9):
  '''
  Computes highest density interval from a sample of representative values,
  estimated as shortest credible interval.
  Arguments:
      Parametrs: 
        - sample_vec: a vector of representative values from a probability distribution.
        - cred_mass: a scalar between 0 and 1, indicating the mass within the credible
          interval that is to be estimated.
      Returns:
        - a vector containing the limits of the hdi
  '''
  sorted_pts = sorted(sample_vec)
  ci_idx_inc = ceil(cred_mass * len(sorted_pts))
  n_cis = len(sorted_pts) - ci_idx_inc
  ci_width = np.zeros(n_cis).tolist()
  for i in range(n_cis):
    ci_width[i] = sorted_pts[i + ci_idx_inc] - sorted_pts[i]
  hdi_min = sorted_pts[np.argmin(ci_width)]
  hdi_max = sorted_pts[np.argmin(ci_width) + ci_idx_inc]
  hdi_lim = [hdi_min, hdi_max]
  
  return hdi_lim

def hdi_of_icdf(icdf_name, cred_mass = 0.95, tol = 1e-8, **args):
    '''
    Calculate HDI from an inverse cumulative density function.
    Implementation from https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region

    Parameters:
        - icdf_name: the ICDF Python function
        - cred_mass: the desired mass of the HDI region
        - tol: tol is passed to the optimize's fmin function
        - args: named parameters related to icdf_name
    Returns: 
        - A list with the HDI limits
    '''
    # freeze distribution with given arguments
    distri = icdf_name(**args)
    # initial guess for HDIlowTailPr
    incred_mass =  1.0 - cred_mass

    def interval_width(low_tail_pr):
        return distri.ppf(cred_mass + low_tail_pr) - distri.ppf(low_tail_pr)

    # find low_tail_pr that minimizes interval_width
    hdi_low_tail_pr = fmin(interval_width, incred_mass, ftol = tol, disp = False)[0]
    # return interval as array([low, high])
    return distri.ppf([hdi_low_tail_pr, cred_mass + hdi_low_tail_pr]).tolist()

def hdi_of_grid(prob_mass_vector, cred_mass = 0.95):
    '''
    Appoximate HDI estimation from a grid

    Parameters:
        - prob_mass_vector: array of probability masses at each grid point
        - cred_mass: the desired mass of the HDI region
    Returns:
        - A list with: a list of the indices in the HDI, the total mass of the
          included indices, the smallest component probability mass in the HDI
    '''
    sorted_prob_mass = np.argsort(-prob_mass_vector)
    hdi_height_idx = np.where(np.cumsum(prob_mass_vector[sorted_prob_mass]) >= cred_mass)[0][0]
    hdi_height = prob_mass_vector[sorted_prob_mass[hdi_height_idx]]
    hdi_indexes = np.where(prob_mass_vector >= hdi_height)[0]
    hdi_mass = np.sum(prob_mass_vector[hdi_indexes])
    return [hdi_indexes, hdi_mass, hdi_height]

#------------------------------------------------------------------------------
# Function(s) for plotting properties of mcmc trace objects.
def dbda_acf_plot(trace, par_name, ax):
    n = int(len(trace[par_name]) / trace.nchains)
    traces = np.array([trace[par_name][i:i + n] for i in range(0, len(trace[par_name]), n)])
    for t in traces:
        ax.plot(acf(t, fft = False), linestyle = 'solid', marker = 'o')
    ax.axhline(y = 0, color = 'black')
    ax.set_ylabel('autocorrelation')
    ax.set_xlabel('lag')
    ax.annotate('ESS = ' + str(round(effective_size(trace[par_name]), 1)), xy = (0.75, 0.85), xycoords = 'axes fraction')

def dbda_dens_plot(trace, par_name, ax, cred_mass = 0.95, range_0_1 = False):
    n = int(len(trace[par_name]) / trace.nchains)
    traces = np.array([trace[par_name][i:i + n] for i in range(0, len(trace[par_name]), n)])
    hdis = []
    for t in traces:
        color = next(ax._get_lines.prop_cycler)['color']
        if range_0_1:
          hdi_99 = [0, 1]
        else:
          hdi_99 = hdi_of_mcmc(t, 0.999)
        x = np.linspace(hdi_99[0], hdi_99[1], 1000)
        ax.plot(x, gaussian_kde(t)(x), color = color)
        hdi = hdi_of_mcmc(t, cred_mass)
        ax.plot(hdi, [0, 0], marker = '|', linestyle = 'None', color = color)
        hdis.append(hdi)
    ax.text(np.mean(hdis), ax.get_ylim()[1] / 10.0, '95% HDI', ha = 'center')
    ax.axhline(y = 0, color = 'black')
    ax.set_ylabel('density')
    ax.set_xlabel('param. value')
    eff_chn_lngth = effective_size(trace[par_name])
    mcse = np.std(traces) / np.sqrt(eff_chn_lngth)
    ax.annotate('MCSE = \n' + str(round(mcse, 5)), 
                xy = (0.85, 0.75), 
                xycoords = 'axes fraction')

def diag_mcmc(trace, par_name = None, range_0_1 = False):
    # This has to be done because I couldn't find how to initialise a parameter
    # from another in Python, as it is done in the R code
    if par_name is None:
        par_name = trace.varnames[-1]

    fig, ax = plt.subplots(2, 2)
    # Traceplot
    n = int(len(trace[par_name]) / trace.nchains)
    traces = np.array([trace[par_name][i:i + n] for i in range(0, len(trace[par_name]), n)])
    for t in traces:
        ax[0][0].plot(t, alpha = 0.5)
    ax[0][0].set_ylabel('param. value')
    ax[0][0].set_xlabel('iteration')
    # Unfortunately I have to do the Gelman plot by hand. Additionally, the function does 
    # not provide quantiles as in the case of the R equivalent. 
    gelman_rubin = []
    for it in range(np.shape(traces)[1]):
        traces_it = traces[:, :it]
        # Code extracted from https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py
        m, n = np.shape(traces_it)
        # Calculate between-chain variance
        B_over_n = np.sum((np.mean(traces_it, 1) - np.mean(traces_it)) ** 2) / (m - 1)
        # Calculate within-chain variances
        W = np.sum([(traces_it[i] - xbar) ** 2 for i, xbar in enumerate(np.mean(traces_it, 1))]) / (m * (n - 1))
        # (over) estimate of variance
        s2 = W * (n - 1) / n + B_over_n
        # Pooled posterior variance estimate
        V = s2 + B_over_n / m
        # Calculate PSRF
        R = V / W

        gelman_rubin.append(np.sqrt(R))

    ax[1][0].plot(range(np.shape(traces)[1]), gelman_rubin)
    ax[1][0].axhline(y = 1, color = 'black')
    ax[1][0].set_xlim([0, np.shape(traces)[1]])
    ax[1][0].set_ylim([np.nanmin(gelman_rubin) - 0.005, np.nanmax(gelman_rubin) + 0.005])
    ax[1][0].set_ylabel('shrink factor')
    ax[1][0].set_xlabel('last iteration in chain')

    dbda_acf_plot(trace, par_name, ax[0][1])
    dbda_dens_plot(trace, par_name, ax[1][1], range_0_1 = range_0_1)

    fig.set_figwidth(12)
    fig.set_figheight(6)

#------------------------------------------------------------------------------
# Functions for summarizing and plotting distribution of a large sample; 
# typically applied to MCMC posterior.

def summarize_post(param_sample_vec, comp_val = None, rope = None, cred_mass = 0.95):
    mean_param = np.mean(param_sample_vec)
    median_param = np.median(param_sample_vec)
    dres = gaussian_kde(param_sample_vec, 2)
    param_l = np.linspace(0, 1, 1000)
    mode_param = param_l[np.argmax(dres(param_l))]
    mcmc_eff_sz = round(effective_size(param_sample_vec), 1)
    hdi_lim = hdi_of_mcmc(param_sample_vec, cred_mass = cred_mass)
    if comp_val is not None:
        pcgt_comp_val = (100 * np.sum(np.array(param_sample_vec) > comp_val) / len(param_sample_vec))
    else:
        pcgt_comp_val = None
    if rope is not None:
        pclt_rope = (100 * np.sum(np.array(param_sample_vec) < rope[0]) / len(param_sample_vec))
        pcgt_rope = (100 * np.sum(np.array(param_sample_vec) > rope[1]) / len(param_sample_vec))
        pcin_rope = 100 - (pclt_rope + pcgt_rope)
    else:
        pclt_rope = None
        pcgt_rope = None
        pcin_rope = None
        rope = [None, None]
    return {'mean': mean_param,
            'median': median_param,
            'mode': mode_param,
            'ess': mcmc_eff_sz,
            'hdi_mass': cred_mass,
            'hdi_low': hdi_lim[0],
            'hdi_high': hdi_lim[1],
            'comp_val': comp_val,
            'pcnt_gt_comp_val': pcgt_comp_val,
            'rope_low': rope[0],
            'rope_high': rope[1],
            'pcnt_lt_rope': pclt_rope,
            'pcnt_in_rope': pcin_rope,
            'pcnt_gt_rope': pcgt_rope}

def plot_post(param_sample_vec,
              cen_tend = None, 
              comp_val = None,
              rope = None,
              cred_mass = 0.95,
              hdi_text_place = 0.7,
              xlab = None,
              xlim = None,
              ylab = None,
              main = None,
              col = None,
              border = None,
              show_curve = False,
              breaks = None,
              ax = None, 
              **kwargs):

  # Override deffaults of hist function, if not specified by the user
  # (additional kwargs arguments are passed to the hist function)
  if xlab is None: 
    xlab = "Param. Val."
  if xlim is None:
    values = np.hstack([v for v in [comp_val, rope, param_sample_vec] if v is not None])
    xlim = [min(values), max(values)]
  if main is None:
    main = ''
  if ylab is None:
    ylab = ''
  if col is None:
    col = 'C0'
  if border is None:
    border = 'w'
  if ax is None:
    fig, ax = plt.subplots()
  
  '''
  # convert coda object to matrix:
  if ( class(paramSampleVec) == "mcmc.list" ) {
    paramSampleVec = as.matrix(paramSampleVec)
  }
  
  '''
  summary_col_names = ['ess',
                       'mean',
                       'median',
                       'mode',
                       'hdi_mass',
                       'hdi_low',
                       'hdi_high',
                       'comp_val',
                       'p_gt_comp_val',
                       'rope_low',
                       'rope_high',
                       'p_lt_rope', 
                       'p_in_rope',
                       'p_gt_rope']
  post_summary = dict((k, None) for k in summary_col_names)

  
  # require(coda) # for effectiveSize function
  post_summary['ess'] = effective_size(param_sample_vec)
  
  post_summary['mean'] = np.mean(param_sample_vec)
  post_summary['median'] = np.median(param_sample_vec)
  mcmc_density = gaussian_kde(param_sample_vec, 2)
  param_l = np.linspace(0, 1, 1000)
  post_summary['mode'] = param_l[np.argmax(mcmc_density(param_l))]
  hdi = hdi_of_mcmc(param_sample_vec, cred_mass)
  post_summary['hdi_mass'] = cred_mass
  post_summary['hdi_low'] = hdi[0]
  post_summary['hdi_high'] = hdi[1]

  # Plot histogram
  cv_col = 'darkgreen'
  rope_col = 'darkred'
  if breaks is None:
    if max(param_sample_vec) > min(param_sample_vec):
      breaks = np.append(np.arange(min(param_sample_vec), 
                                   max(param_sample_vec),
                                   (hdi[1] - hdi[0]) / 18.0), 
                         max(param_sample_vec)).tolist()
    else:
      breaks = [min(param_sample_vec) - 1.0e-6, max(param_sample_vec) + 1.0e-6]
      border = 'C0'

  if not show_curve:
    histinfo, _, _ = ax.hist(param_sample_vec, 
                             bins = breaks, 
                             edgecolor = border,
                             facecolor = col,
                             density = 1)
  if show_curve:
    histinfo, _ = np.histogram(param_sample_vec, 
                               bins = breaks,
                               weights = np.ones_like(param_sample_vec) / \
                                         len(param_sample_vec))
    
    density = gaussian_kde(param_sample_vec)
    ax.plot(param_l, density(param_l), lw = 3, color = col)

  ax.set_title(main)
  ax.set_xlabel(xlab)
  ax.set_ylabel(ylab)
  ax.set_xlim(xlim)

  cen_tend_ht = 0.9 * max(histinfo)
  cv_ht = 0.7 * max(histinfo)
  rope_text_ht = 0.55 * max(histinfo)
  # Display central tendency:
  mn = np.mean(param_sample_vec)
  med = np.median(param_sample_vec)
  mcmc_density = gaussian_kde(param_sample_vec)
  mo = post_summary['mode']
  if cen_tend == 'mode':
    ax.text(mo, cen_tend_ht, 'mode = ' + str(round(mo, 3)), ha = 'center')
  if cen_tend == 'median':
    ax.text(med, cen_tend_ht, 'median = ' + str(round(med, 3)), ha = 'center')
  if cen_tend == 'mean':
    ax.text(mn, cen_tend_ht, 'mean = ' + str(round(mo, 3)), ha = 'center')
  # Display the comparison value.
  if comp_val is not None:
    p_gt_comp_val = np.sum(param_sample_vec > comp_val) / len(param_sample_vec)
    p_lt_comp_val = 1 - p_gt_comp_val
    ax.axvline(comp_val, ls = ':', lw = 2, color = cv_col)
    ax.text(comp_val, cv_ht, str(round(100 * p_lt_comp_val, 1)) + \
                             '% < ' + \
                             str(round(comp_val, 3)) + \
                             ' < ' + \
                             str(round(100 * p_gt_comp_val, 1)) + \
                             '%', ha = 'center')
    post_summary['comp_val'] = comp_val
    post_summary['p_gt_comp_val'] = p_gt_comp_val
  # Display the ROPE.
  if rope is not None:
    p_in_rope = np.sum(np.logical_and(param_sample_vec > rope[0],
                                      param_sample_vec < rope[1])) / len(param_sample_vec)
    p_gt_rope = np.sum(param_sample_vec >= rope[1]) / len(param_sample_vec)
    p_lt_rope = np.sum(param_sample_vec <= rope[0]) / len(param_sample_vec)
    ax.axvline(rope[0], ls = ':', lw = 2, color = rope_col)
    ax.axvline(rope[1], ls = ':', lw = 2, color = rope_col)
    ax.text(np.mean(rope), rope_text_ht, str(round(100 * p_lt_rope, 1)) + \
                                         '% < ' + \
                                         str(rope[0]) + \
                                         ' < ' + \
                                         str(round(p_in_rope, 1)) + \
                                         '% < ' + \
                                         str(rope[1]) + \
                                         ' < ' + \
                                         str(round(100 * p_gt_rope, 1)) + \
                                         '%', ha = 'center')
    post_summary['rope_low'] = rope[0]
    post_summary['rope_high'] = rope[1]
    post_summary['p_lt_rope'] = p_lt_rope
    post_summary['p_in_rope'] = p_in_rope
    post_summary['p_gt_rope'] = p_gt_rope
  # Display the HDI
  ax.axvline(hdi[0], lw = 4, color = 'C1')
  ax.axvline(hdi[1], lw = 4, color = 'C1')
  ax.text(np.mean(hdi), 0, str(100 * cred_mass) + '% HDI', ha = 'center')
  ax.text(hdi[0], 0, str(round(hdi[0], 3)), ha = 'center')
  ax.text(hdi[1], 0, str(round(hdi[1], 3)), ha = 'center')

  return post_summary
