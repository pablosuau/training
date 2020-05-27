import pandas as pd
from jags_ydich_xnomssubj_mbernbeta import gen_mcmc
from dbda2e_utilities import diag_mcmc

# Example for jags_ydich_xnomssubj_mbernbeta.py

def jags_ydich_xnomssubj_mbernbeta_example(data_csv = 'z6n8z2n7.csv'):
	my_data = pd.read_csv(data_csv)
	
	# Generate the MCMC chain
	mcmc_trace = gen_mcmc(data = my_data, num_saved_steps = 50000)

	# Display diagnostics of chain. for specificed parameters
	parameter_names = [v for v in mcmc_trace.varnames if not 'logodds' in v] # get all parameter names
	for par_name in parameter_names:
		diag_mcmc(mcmc_trace, par_name = par_name)

'''
# Load the relevant model into R's working memory:
source("Jags-Ydich-XnomSsubj-MbernBeta.R")
#------------------------------------------------------------------------------- 
# Optional: Specify filename root and graphical format for saving output.
# Otherwise specify as NULL or leave saveName and saveType arguments 
# out of function calls.
fileNameRoot = "Jags-Ydich-XnomSsubj-MbernBeta-" 
graphFileType = "eps" 
#------------------------------------------------------------------------------- 
# Generate the MCMC chain:
mcmcCoda = genMCMC( data=myData , numSavedSteps=50000 , saveName=fileNameRoot )
#------------------------------------------------------------------------------- 
# Display diagnostics of chain, for specified parameters:
parameterNames = varnames(mcmcCoda) # get all parameter names
for ( parName in parameterNames ) {
  diagMCMC( codaObject=mcmcCoda , parName=parName , 
                saveName=fileNameRoot , saveType=graphFileType )
}
#------------------------------------------------------------------------------- 
# Get summary statistics of chain:
summaryInfo = smryMCMC( mcmcCoda , compVal=NULL , #rope=c(0.45,0.55) ,
                        compValDiff=0.0 , #ropeDiff = c(-0.05,0.05) ,
                        saveName=fileNameRoot )
# Display posterior information:
plotMCMC( mcmcCoda , data=myData , compVal=NULL , #rope=c(0.45,0.55) ,
          compValDiff=0.0 , #ropeDiff = c(-0.05,0.05) ,
          saveName=fileNameRoot , saveType=graphFileType )
#------------------------------------------------------------------------------- 
'''