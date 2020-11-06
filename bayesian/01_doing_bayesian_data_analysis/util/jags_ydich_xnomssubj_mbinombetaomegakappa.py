import numpy as np
import pymc3 as pm
from math import ceil

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

'''
#===============================================================================

smryMCMC = function(  codaSamples , compVal=0.5 , rope=NULL , 
                      diffIdVec=NULL , compValDiff=0.0 , ropeDiff=NULL , 
                      saveName=NULL ) {
  mcmcMat = as.matrix(codaSamples,chains=TRUE)
  Ntheta = length(grep("theta",colnames(mcmcMat)))
  summaryInfo = NULL
  rowIdx = 0
  # overall omega:
  summaryInfo = rbind( summaryInfo , 
                       summarizePost( mcmcMat[,"omega"] ,
                                      compVal=compVal , ROPE=rope ) )
  rowIdx = rowIdx+1
  rownames(summaryInfo)[rowIdx] = "omega"
  # kappa:
  summaryInfo = rbind( summaryInfo , 
                       summarizePost( mcmcMat[,"kappa"] ,
                                      compVal=NULL , ROPE=NULL ) )
  rowIdx = rowIdx+1
  rownames(summaryInfo)[rowIdx] = "kappa"
  # individual theta's:
  for ( tIdx in 1:Ntheta ) {
    parName = paste0("theta[",tIdx,"]")
    summaryInfo = rbind( summaryInfo , 
      summarizePost( mcmcMat[,parName] , compVal=compVal , ROPE=rope ) )
    rowIdx = rowIdx+1
    rownames(summaryInfo)[rowIdx] = parName
  }
  # differences of individual theta's:
  if ( !is.null(diffIdVec) ) {
    Nidx = length(diffIdVec)
    for ( t1Idx in 1:(Nidx-1) ) {
      for ( t2Idx in (t1Idx+1):Nidx ) {
        parName1 = paste0("theta[",diffIdVec[t1Idx],"]")
        parName2 = paste0("theta[",diffIdVec[t2Idx],"]")
        summaryInfo = rbind( summaryInfo , 
          summarizePost( mcmcMat[,parName1]-mcmcMat[,parName2] ,
                         compVal=compValDiff , ROPE=ropeDiff ) )
        rowIdx = rowIdx+1
        rownames(summaryInfo)[rowIdx] = paste0(parName1,"-",parName2)
      }
    }
  }
  # save:
  if ( !is.null(saveName) ) {
    write.csv( summaryInfo , file=paste(saveName,"SummaryInfo.csv",sep="") )
  }
  show( summaryInfo )
  return( summaryInfo )
}

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