import pandas as pd
from jags_ydich_xnomssubj_mbinombetaomegakappa import gen_mcmc
from dbda2e_utilities import diag_mcmc

# Example for jags_ydich_xnomssubj_mbinombetaomegakappa.py

def jags_ydich_xnomssubj_mbinombetaomegakappa_example(data_csv = 'therapeutic_touch_data.csv'):
  my_data = pd.read_csv(data_csv)


	# Generate the MCMC chain
  mcmc_trace = gen_mcmc(data = my_data, 
						  s_name = 's', 
						  y_name = 'y', 
						  num_saved_steps = 20000, 
						  thin_steps = 10)

  # Display diagnostics of chain, for specified parameters
  #parameter_names = [v for v in mcmc_trace.varnames if 'logodds' not in v] # get all parameter names
  #for par_name in parameter_names:
  #  diag_mcmc(mcmc_trace, par_name = par_name)
'''

#------------------------------------------------------------------------------- 
# Display diagnostics of chain, for specified parameters:
parameterNames = varnames(mcmcCoda) # get all parameter names for reference
for ( parName in parameterNames[c(1:3,length(parameterNames))] ) { 
  diagMCMC( codaObject=mcmcCoda , parName=parName , 
                saveName=fileNameRoot , saveType=graphFileType )
}
#------------------------------------------------------------------------------- 
# Get summary statistics of chain:
summaryInfo = smryMCMC( mcmcCoda , compVal=0.5 , 
                        diffIdVec=c(1,14,28), compValDiff=0.0, 
                        saveName=fileNameRoot )
# Display posterior information:
plotMCMC( mcmcCoda , data=myData , sName="s" , yName="y" , 
          compVal=0.5 , #rope=c(0.45,0.55) ,
          diffIdVec=c(1,14,28), compValDiff=0.0, #ropeDiff = c(-0.05,0.05) ,
          saveName=fileNameRoot , saveType=graphFileType )
#------------------------------------------------------------------------------- 
'''