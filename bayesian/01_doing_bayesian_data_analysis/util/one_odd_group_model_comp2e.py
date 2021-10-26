import numpy as np
import pandas as pd
from scipy.stats import binom
#------------------------------------------------------------------------------
# THE DATA.
# Randomly generated fictitous data.
# For each subject, specify the condition s/he was in,
# the number of trials s/he experienced, and the number correct.
npg = 20 # Number of subjects per group
ntrl = 20 # Number of trials per subject
cond_of_subj = np.hstack((np.repeat(0, npg),
                          np.repeat(1, npg),
                          np.repeat(2, npg),
                          np.repeat(3, npg)))
n_trl_of_subj = np.repeat(ntrl, 4 * npg)
np.random.seed(47405)
cond_means = np.array([0.4, 0.5, 0.51, 0.52])
n_corr_of_subj = np.hstack((binom.rvs(ntrl, cond_means[0], size = npg),
                            binom.rvs(ntrl, cond_means[1], size = npg),
                            binom.rvs(ntrl, cond_means[2], size = npg),
                            binom.rvs(ntrl, cond_means[3], size = npg)))
n_cond = len(list(np.unique(cond_of_subj)))
n_subj = len(list(cond_of_subj))
# Jitter the data to be as close as possible to desired condition means:
for c_idx in range(n_cond):
  n_to_add = int(round(cond_means[c_idx] * npg * ntrl - np.sum(n_corr_of_subj[cond_of_subj == c_idx])))
  if n_to_add > 0:
    for i in range(n_to_add):
      this_n_corr = ntrl
      while this_n_corr == ntrl:
        rand_subj_idx = np.random.choice(np.where(cond_of_subj == c_idx)[0])
        this_n_corr = n_corr_of_subj[rand_subj_idx]
      n_corr_of_subj[rand_subj_idx] = n_corr_of_subj[rand_subj_idx] + 1
  if n_to_add < 0:
    for i in range(abs(n_to_add)):
      this_n_corr = 0
      while this_n_corr == 0:
        rand_subj_idx = np.random.choice(np.where(cond_of_subj == c_idx)[0])
        this_n_corr = n_corr_of_subj[rand_subj_idx]
      n_corr_of_subj[rand_subj_idx] = n_corr_of_subj[rand_subj_idx] - 1
print(pd.DataFrame(np.vstack((n_corr_of_subj, cond_of_subj)).T, 
                   columns = ['n_corr_of_subj', 'cond_of_subj']) \
        .groupby('cond_of_subj') \
        .mean() \
        .rename(columns = {'cond_of_subj': 'mean'}) / ntrl)
'''

# Package the data:
dataList = list(
  nCond = nCond ,
  nSubj = nSubj ,
  CondOfSubj = CondOfSubj ,
  nTrlOfSubj = nTrlOfSubj ,
  nCorrOfSubj = nCorrOfSubj
)

#------------------------------------------------------------------------------
# THE MODEL.

modelString = "
model {
  for ( s in 1:nSubj ) {
    nCorrOfSubj[s] ~ dbin( theta[s] , nTrlOfSubj[s] )
    theta[s] ~ dbeta( aBeta[CondOfSubj[s]] , bBeta[CondOfSubj[s]] ) 
  }

#   for ( j in 1:nCond ) {
#     # Use omega[j] for model index 1, omega0 for model index 2:
#     aBeta[j] <-       ( equals(mdlIdx,1)*omega[j] 
#                       + equals(mdlIdx,2)*omega0  )   * (kappa[j]-2)+1
#     bBeta[j] <- ( 1 - ( equals(mdlIdx,1)*omega[j] 
#                       + equals(mdlIdx,2)*omega0  ) ) * (kappa[j]-2)+1
#   }
#   for ( j in 1:2 ) {
#     omega[j] ~ dbeta( a[j,mdlIdx] , b[j,mdlIdx] )
#   }
#   omega[3] <- omega[2]
#   omega[4] <- omega[2]

  for ( j in 1:nCond ) {
    # Use omega[j] for model index 1, omega0 for model index 2:
    aBeta[j] <-       ( equals(mdlIdx,1)*omega[j] 
                      + equals(mdlIdx,2)*omega0  )   * (kappa[j]-2)+1
    bBeta[j] <- ( 1 - ( equals(mdlIdx,1)*omega[j] 
                      + equals(mdlIdx,2)*omega0  ) ) * (kappa[j]-2)+1
    omega[j] ~ dbeta( a[j,mdlIdx] , b[j,mdlIdx] )
  }

  omega0 ~ dbeta( a0[mdlIdx] , b0[mdlIdx] )
  for ( j in 1:nCond ) {
    kappa[j] <- kappaMinusTwo[j] + 2
    kappaMinusTwo[j] ~ dgamma( 2.618 , 0.0809 ) # mode 20 , sd 20
  }
  # Constants for prior and pseudoprior:
  aP <- 1
  bP <- 1
  # a0[model] and b0[model]
  a0[1] <- .48*500       # pseudo
  b0[1] <- (1-.48)*500   # pseudo 
  a0[2] <- aP            # true
  b0[2] <- bP            # true
  # a[condition,model] and b[condition,model]
  a[1,1] <- aP           # true
  a[2,1] <- aP           # true
  a[3,1] <- aP           # true
  a[4,1] <- aP           # true
  b[1,1] <- bP           # true
  b[2,1] <- bP           # true
  b[3,1] <- bP           # true
  b[4,1] <- bP           # true
  a[1,2] <- .40*125      # pseudo
  a[2,2] <- .50*125      # pseudo
  a[3,2] <- .51*125      # pseudo
  a[4,2] <- .52*125      # pseudo
  b[1,2] <- (1-.40)*125  # pseudo
  b[2,2] <- (1-.50)*125  # pseudo
  b[3,2] <- (1-.51)*125  # pseudo
  b[4,2] <- (1-.52)*125  # pseudo
  # Prior on model index:
  mdlIdx ~ dcat( modelProb[] )
  modelProb[1] <- .5
  modelProb[2] <- .5
}
" # close quote for modelstring
writeLines( modelString , con="TEMPmodel.txt" )

#------------------------------------------------------------------------------
# INTIALIZE THE CHAINS.

# Let JAGS do it...

#------------------------------------------------------------------------------
# RUN THE CHAINS.

parameters = c("omega","kappa","omega0","theta","mdlIdx")
adaptSteps = 1000            # Number of steps to "tune" the samplers.
burnInSteps = 5000           # Number of steps to "burn-in" the samplers.
nChains = 3                  # Number of chains to run.
numSavedSteps=12000          # Total number of steps in chains to save.
thinSteps=20                 # Number of steps to "thin" (1=keep every step).

# nPerChain = ceiling( ( numSavedSteps * thinSteps ) / nChains ) # Steps per chain.
# # Create, initialize, and adapt the model:
# jagsModel = jags.model( "TEMPmodel.txt" , data=dataList , # inits=initsList , 
#                         n.chains=nChains , n.adapt=adaptSteps )
# # Burn-in:
# cat( "Burning in the MCMC chain...\n" )
# update( jagsModel , n.iter=burnInSteps )
# # The saved MCMC chain:
# cat( "Sampling final MCMC chain...\n" )
# codaSamples = coda.samples( jagsModel , variable.names=parameters , 
#                             n.iter=nPerChain , thin=thinSteps )

runJagsOut <- run.jags( method=c("rjags","parallel")[2] ,
                        model="TEMPmodel.txt" , 
                        monitor=parameters , 
                        data=dataList ,  
                        #inits=initsList , 
                        n.chains=nChains ,
                        adapt=adaptSteps ,
                        burnin=burnInSteps , 
                        sample=ceiling(numSavedSteps/nChains) ,
                        thin=thinSteps ,
                        summarise=FALSE ,
                        plots=FALSE )
codaSamples = as.mcmc.list( runJagsOut )

# resulting codaSamples object has these indices: 
#   codaSamples[[ chainIdx ]][ stepIdx , paramIdx ]

save( codaSamples , file=paste0(fileNameRoot,"Mcmc.Rdata") )

#------------------------------------------------------------------------------- 
# Display diagnostics of chain:

parameterNames = varnames(codaSamples) # get all parameter names
show(parameterNames)
for ( parName in c("mdlIdx","omega[1]","omega0","kappa[1]","theta[1]") ) { 
  diagMCMC( codaSamples , parName=parName ,
            saveName=fileNameRoot , saveType="eps" )
}

#------------------------------------------------------------------------------
# EXAMINE THE RESULTS.

mcmcMat = as.matrix(codaSamples,chains=TRUE)

xLim=c(0.35,0.75)

# Display the model index
modelIdxSample = mcmcMat[, "mdlIdx" ]
pM1 = sum( modelIdxSample == 1 ) / length( modelIdxSample )
pM2 = 1 - pM1
string1 =paste("p( Diff Omega M1 | D )=",round(pM1,3),sep="")
string2 =paste("p( Same Omega M2 | D )=",round(pM2,3),sep="")
openGraph(10,4)
nStepsToPlot = 1000
plot( 1:nStepsToPlot , modelIdxSample[1:nStepsToPlot] , type="l" , lwd=2 ,
      xlab="Step in Markov chain" , ylab="Model Index (1, 2)" ,
      main=paste(string1,", ",string2,sep="") , col="skyblue" )
saveGraph(file=paste0(fileNameRoot,"MdlIdx"),type="eps")

# Display the omega0 posterior
omega0sampleM1 = mcmcMat[, "omega0" ][ modelIdxSample == 1 ]
omega0sampleM2 = mcmcMat[, "omega0" ][ modelIdxSample == 2 ]
openGraph()
layout( matrix(1:2,nrow=2) )
plotPost( omega0sampleM1 , main="Pseudoprior for M = 1 (Diff Omega)" ,
      xlab=expression(omega[0]) , xlim=xLim )
plotPost( omega0sampleM2 , main="Posterior for M = 2 (Same Omega)"  ,
      xlab=expression(omega[0]) , xlim=xLim )
saveGraph(file=paste0(fileNameRoot,"Omega0"),type="eps")

# Display the omega[j] posterior
omega1sampleM1 = mcmcMat[, "omega[1]" ][ modelIdxSample == 1 ]
omega2sampleM1 = mcmcMat[, "omega[2]" ][ modelIdxSample == 1 ]
omega3sampleM1 = mcmcMat[, "omega[3]" ][ modelIdxSample == 1 ]
omega4sampleM1 = mcmcMat[, "omega[4]" ][ modelIdxSample == 1 ]
omega1sampleM2 = mcmcMat[, "omega[1]" ][ modelIdxSample == 2 ]
omega2sampleM2 = mcmcMat[, "omega[2]" ][ modelIdxSample == 2 ]
omega3sampleM2 = mcmcMat[, "omega[3]" ][ modelIdxSample == 2 ]
omega4sampleM2 = mcmcMat[, "omega[4]" ][ modelIdxSample == 2 ]
openGraph(10,5)
layout( matrix(1:8,nrow=2,byrow=T) )
plotPost( omega1sampleM1 , main="Posterior for M = 1 (Diff Omega)" ,
          xlab=expression(omega[1]) , xlim=xLim )
plotPost( omega2sampleM1 , main="Posterior for M = 1 (Diff Omega)" ,
          xlab=expression(omega[2]) , xlim=xLim )
plotPost( omega3sampleM1 , main="Posterior for M = 1 (Diff Omega)" ,
          xlab=expression(omega[3]) , xlim=xLim )
plotPost( omega4sampleM1 , main="Posterior for M = 1 (Diff Omega)" ,
          xlab=expression(omega[4]) , xlim=xLim )
plotPost( omega1sampleM2 , main="Pseudoprior for M = 2 (Same Omega)" ,
          xlab=expression(omega[1]) , xlim=xLim )
plotPost( omega2sampleM2 , main="Pseudoprior for M = 2 (Same Omega)" ,
          xlab=expression(omega[2]) , xlim=xLim )
plotPost( omega3sampleM2 , main="Pseudoprior for M = 2 (Same Omega)" ,
          xlab=expression(omega[3]) , xlim=xLim )
plotPost( omega4sampleM2 , main="Pseudoprior for M = 2 (Same Omega)" ,
          xlab=expression(omega[4]) , xlim=xLim )
saveGraph(file=paste0(fileNameRoot,"OmegaCond"),type="eps")


# Display the differences of omega[j]'s
omegaSample = rbind( omega1sampleM1 , omega2sampleM1 , omega3sampleM1 , omega4sampleM1 )
openGraph(10,5)
layout( matrix(1:6,nrow=2,ncol=3,byrow=T) )
xmin = -0.25
xmax = 0.25
for ( i in 1:3 ) {
    for ( j in (i+1):4 ) {
        plotPost( omegaSample[i,]-omegaSample[j,] , compVal=0.0 ,
                  xlab=bquote(omega[.(i)]-omega[.(j)]) ,
                  #breaks=unique( c( min(c(xmin,omegaSample[i,]-omegaSample[j,])),
                  #          seq(xmin,xmax,len=20),
                  #          max(c(xmax,omegaSample[i,]-omegaSample[j,])) )) ,
                  main="" , xlim=c(xmin,xmax) )
    }
}
saveGraph(file=paste0(fileNameRoot,"OmegaDiff"),type="eps")
'''