import numpy as np
import pandas as pd
import pymc3 as pm
from math import ceil
from dbda2e_utilities import summarize_post

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

'''

#===============================================================================

plotMCMC = function( codaSamples , data , sName="s" , yName="y" , 
                     compVal=0.5 , rope=NULL , 
                     diffIdVec=NULL , compValDiff=0.0 , ropeDiff=NULL , 
                     saveName=NULL , saveType="jpg" ) {
  #-----------------------------------------------------------------------------
  # N.B.: This function expects the data to be a data frame, 
  # with one component named y being a vector of integer 0,1 values,
  # and one component named s being a factor of subject identifiers.
  y = data[,yName]
  s = as.numeric(data[,sName]) # ensures consecutive integer levels
  # Now plot the posterior:
  mcmcMat = as.matrix(codaSamples,chains=TRUE)
  chainLength = NROW( mcmcMat )
  # Plot omega, kappa:
  openGraph(width=3.5*2,height=3.0)
  par( mfrow=c(1,2) )
  par( mar=c(3.5,3,1,1) , mgp=c(2.0,0.7,0) )
  postInfo = plotPost( mcmcMat[,"kappa"] , compVal=NULL , ROPE=NULL ,
                       xlab=bquote(kappa) , main="" , 
                       xlim=c( min(mcmcMat[,"kappa"]),
                               quantile(mcmcMat[,"kappa"],probs=c(0.990)) ) )
  postInfo = plotPost( mcmcMat[,"omega"] , compVal=compVal , ROPE=rope ,
                       xlab=bquote(omega) , main="Group Mode" ,
                       xlim=quantile(mcmcMat[,"omega"],probs=c(0.005,0.995)) )
  if ( !is.null(saveName) ) {
    saveGraph( file=paste(saveName,"PostOmega",sep=""), type=saveType)
  }
  # Plot individual theta's and differences:
  if ( !is.null(diffIdVec) ) {
    Nidx = length(diffIdVec)
    openGraph(width=2.5*Nidx,height=2.0*Nidx)
    par( mfrow=c(Nidx,Nidx) )
    for ( t1Idx in 1:Nidx ) {
      for ( t2Idx in 1:Nidx ) {
        parName1 = paste0("theta[",diffIdVec[t1Idx],"]")
        parName2 = paste0("theta[",diffIdVec[t2Idx],"]")
        if ( t1Idx > t2Idx) {  
          # plot.new() # empty plot, advance to next
          par( mar=c(3.5,3.5,1,1) , mgp=c(2.0,0.7,0) , pty="s" )
          nToPlot = 700
          ptIdx = round(seq(1,chainLength,length=nToPlot))
          plot ( mcmcMat[ptIdx,parName2] , mcmcMat[ptIdx,parName1] , cex.lab=1.75 ,
                 xlab=parName2 , ylab=parName1 , col="skyblue" )
          abline(0,1,lty="dotted")
        } else if ( t1Idx == t2Idx ) {
          par( mar=c(3.5,1,1,1) , mgp=c(2.0,0.7,0) , pty="m" )
          postInfo = plotPost( mcmcMat[,parName1] , cex.lab = 1.75 , 
                               compVal=compVal , ROPE=rope , cex.main=1.5 ,
                               xlab=parName1 , main="" )
          includeRows = ( s == diffIdVec[t1Idx] ) # rows of this subject in data
          dataPropor = sum(y[includeRows])/sum(includeRows) 
          points( dataPropor , 0 , pch="+" , col="red" , cex=3 )
        } else if ( t1Idx < t2Idx ) {
          par( mar=c(3.5,1,1,1) , mgp=c(2.0,0.7,0) , pty="m" )
          postInfo = plotPost( mcmcMat[,parName1]-mcmcMat[,parName2] , 
                            compVal=compValDiff , ROPE=ropeDiff , cex.main=1.5 ,
                            xlab=paste0(parName1,"-",parName2) , main="" , 
                            cex.lab=1.75 )
          includeRows1 = ( s == diffIdVec[t1Idx] ) # rows of this subject in data
          dataPropor1 = sum(y[includeRows1])/sum(includeRows1) 
          includeRows2 = ( s == diffIdVec[t2Idx] ) # rows of this subject in data
          dataPropor2 = sum(y[includeRows2])/sum(includeRows2) 
          points( dataPropor1-dataPropor2 , 0 , pch="+" , col="red" , cex=3 )
        }
      }
    }
  }
  if ( !is.null(saveName) ) {
    saveGraph( file=paste(saveName,"PostTheta",sep=""), type=saveType)
  }
}

#===============================================================================
'''