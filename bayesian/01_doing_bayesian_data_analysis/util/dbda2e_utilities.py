import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.optimize import fmin
from scipy.stats import *
from statsmodels.tsa.stattools import acf

#------------------------------------------------------------------------------
# Implementation of R functions not in Python
def effective_size(x):
    # Where x is a list
    acf_x = acf(x)
    acf_x = [f for f in acf_x if f >= 0.05]
    
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
  hdi_max = sorted_pts[np.argmax(ci_width) + ci_idx_inc]
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
# Functions for summarizing and plotting distribution of a large sample; 
# typically applied to MCMC posterior.

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
    values = [comp_val, rope, param_sample_vec]
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
                       'pLtROPE', 
                       'pInROPE',
                       'pGtROPE']
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


  print(post_summary)
  '''
  

  cenTendHt = 0.9*max(histinfo$density)
  cvHt = 0.7*max(histinfo$density)
  ROPEtextHt = 0.55*max(histinfo$density)
  # Display central tendency:
  mn = mean(paramSampleVec)
  med = median(paramSampleVec)
  mcmcDensity = density(paramSampleVec)
  mo = mcmcDensity$x[which.max(mcmcDensity$y)]
  if ( cenTend=="mode" ){ 
    text( mo , cenTendHt ,
          bquote(mode==.(signif(mo,3))) , adj=c(.5,0) ,  )
  }
  if ( cenTend=="median" ){ 
    text( med , cenTendHt ,
          bquote(median==.(signif(med,3))) , adj=c(.5,0) , , col=cvCol )
  }
  if ( cenTend=="mean" ){ 
    text( mn , cenTendHt ,
          bquote(mean==.(signif(mn,3))) , adj=c(.5,0) , )
  }
  # Display the comparison value.
  if ( !is.null( compVal ) ) {
    pGtCompVal = sum( paramSampleVec > compVal ) / length( paramSampleVec ) 
    pLtCompVal = 1 - pGtCompVal
    lines( c(compVal,compVal) , c(0.96*cvHt,0) , 
           lty="dotted" , lwd=2 , col=cvCol )
    text( compVal , cvHt ,
          bquote( .(round(100*pLtCompVal,1)) * "% < " *
                   .(signif(compVal,3)) * " < " * 
                   .(round(100*pGtCompVal,1)) * "%" ) ,
          adj=c(pLtCompVal,0) ,  , col=cvCol )
    postSummary[,"compVal"] = compVal
    postSummary[,"pGtCompVal"] = pGtCompVal
  }
  # Display the ROPE.
  if ( !is.null( ROPE ) ) {
    pInROPE = ( sum( paramSampleVec > ROPE[1] & paramSampleVec < ROPE[2] )
                / length( paramSampleVec ) )
    pGtROPE = ( sum( paramSampleVec >= ROPE[2] ) / length( paramSampleVec ) )
    pLtROPE = ( sum( paramSampleVec <= ROPE[1] ) / length( paramSampleVec ) )
    lines( c(ROPE[1],ROPE[1]) , c(0.96*ROPEtextHt,0) , lty="dotted" , lwd=2 ,
           col=ropeCol )
    lines( c(ROPE[2],ROPE[2]) , c(0.96*ROPEtextHt,0) , lty="dotted" , lwd=2 ,
           col=ropeCol)
    text( mean(ROPE) , ROPEtextHt ,
          bquote( .(round(100*pLtROPE,1)) * "% < " * .(ROPE[1]) * " < " * 
                   .(round(100*pInROPE,1)) * "% < " * .(ROPE[2]) * " < " * 
                   .(round(100*pGtROPE,1)) * "%" ) ,
          adj=c(pLtROPE+.5*pInROPE,0) ,  , col=ropeCol )
    
    postSummary[,"ROPElow"]=ROPE[1] 
    postSummary[,"ROPEhigh"]=ROPE[2] 
    postSummary[,"pLtROPE"]=pLtROPE
    postSummary[,"pInROPE"]=pInROPE
    postSummary[,"pGtROPE"]=pGtROPE
  }
  # Display the HDI.
  lines( HDI , c(0,0) , lwd=4 , lend=1 )
  text( mean(HDI) , 0 , bquote(.(100*credMass) * "% HDI" ) ,
        adj=c(.5,-1.7) ,  )
  text( HDI[1] , 0 , bquote(.(signif(HDI[1],3))) ,
        adj=c(HDItextPlace,-0.5) ,  )
  text( HDI[2] , 0 , bquote(.(signif(HDI[2],3))) ,
        adj=c(1.0-HDItextPlace,-0.5) ,  )
  par(xpd=F)
  #

  return( postSummary )
}
  '''
