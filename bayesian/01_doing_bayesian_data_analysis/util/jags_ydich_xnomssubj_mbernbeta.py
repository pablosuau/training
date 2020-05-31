import numpy as np
import pandas as pd
import pymc3 as pm
from math import ceil
from dbda2e_utilities import summarize_post

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
    parameters = [] # The parameters to be monutored
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
    print(summary_info)
    return(summary_info)

def plot_mcmc(trace, data, comp_val = 0.5, rope = None, comp_val_diff = 0, rope_diff = None):
    pass

'''
#===============================================================================

plotMCMC = function( codaSamples , data , compVal=0.5 , rope=NULL , 
                     compValDiff=0.0 , ropeDiff=NULL , 
                     saveName=NULL , saveType="jpg" ) {
  #-----------------------------------------------------------------------------
  # N.B.: This function expects the data to be a data frame, 
  # with one component named y being a vector of integer 0,1 values,
  # and one component named s being a factor of subject identifiers.
  y = data$y
  s = as.numeric(data$s) # converts character to consecutive integer levels
  # Now plot the posterior:
  mcmcMat = as.matrix(codaSamples,chains=TRUE)
  chainLength = NROW( mcmcMat )
  Ntheta = length(grep("theta",colnames(mcmcMat)))
  openGraph(width=2.5*Ntheta,height=2.0*Ntheta)
  par( mfrow=c(Ntheta,Ntheta) )
  for ( t1Idx in 1:(Ntheta) ) {
    for ( t2Idx in (1):Ntheta ) {
      parName1 = paste0("theta[",t1Idx,"]")
      parName2 = paste0("theta[",t2Idx,"]")
      if ( t1Idx > t2Idx) {  
        # plot.new() # empty plot, advance to next
        par( mar=c(3.5,3.5,1,1) , mgp=c(2.0,0.7,0) )
        nToPlot = 700
        ptIdx = round(seq(1,chainLength,length=nToPlot))
        plot ( mcmcMat[ptIdx,parName2] , mcmcMat[ptIdx,parName1] , cex.lab=1.75 ,
               xlab=parName2 , ylab=parName1 , col="skyblue" )
      } else if ( t1Idx == t2Idx ) {
        par( mar=c(3.5,1,1,1) , mgp=c(2.0,0.7,0) )
        postInfo = plotPost( mcmcMat[,parName1] , cex.lab = 1.75 , 
                             compVal=compVal , ROPE=rope , cex.main=1.5 ,
                             xlab=parName1 , main="" )
        includeRows = ( s == t1Idx ) # identify rows of this subject in data
        dataPropor = sum(y[includeRows])/sum(includeRows) 
        points( dataPropor , 0 , pch="+" , col="red" , cex=3 )
      } else if ( t1Idx < t2Idx ) {
        par( mar=c(3.5,1,1,1) , mgp=c(2.0,0.7,0) )
        postInfo = plotPost(mcmcMat[,parName1]-mcmcMat[,parName2] , cex.lab = 1.75 , 
                           compVal=compValDiff , ROPE=ropeDiff , cex.main=1.5 ,
                           xlab=paste0(parName1,"-",parName2) , main="" )
        includeRows1 = ( s == t1Idx ) # identify rows of this subject in data
        dataPropor1 = sum(y[includeRows1])/sum(includeRows1) 
        includeRows2 = ( s == t2Idx ) # identify rows of this subject in data
        dataPropor2 = sum(y[includeRows2])/sum(includeRows2) 
        points( dataPropor1-dataPropor2 , 0 , pch="+" , col="red" , cex=3 )
      }
    }
  }
  #-----------------------------------------------------------------------------  
  if ( !is.null(saveName) ) {
    saveGraph( file=paste(saveName,"Post",sep=""), type=saveType)
  }
}

#===============================================================================
'''