import pandas as pd
import pymc3 as pm
from dbda2e_utilities import diag_mcmc

# Load the data (it has to be in the current working directory)
my_data = pd.read_csv('z15n50.csv')
y = my_data.y.values # The y values are in the column named y
n_total = y.shape[0]

# Define the model
# PyMC3 uses a different model by default. Therefore, most of the initialisation and additional parameters
# in the R code are not relevant in the Python version. For instance, we do not need an update(n.iter = 500) 
# equivalent since the resulting traces already have their burn-in period already filtered out, and it consists
# of 500 steps
with pm.Model() as model:
	theta = pm.Beta('theta', alpha = 1, beta = 1)
	y_obs = pm.Bernoulli('y_obs', p = theta, observed = y)
	# Run the chains - no need to add jitter since PyMC3 already does it
    # Other values used in the R code are set by default in PyMC3
	trace = pm.sample(chains = 3, draws = 3334)
    
# Examine the chains
# Convergence diagnostics
diag_mcmc(trace, par_name = 'theta')
print(len(trace['theta'])) # Remove this
print(trace['theta']) # Remove this
print(dir(trace))
print(trace.varnames)


'''

# Examine the chains: - codaSamples is my trace in Python
# Convergence diagnostics:
diagMCMC( codaObject=codaSamples , parName="theta" )
saveGraph( file=paste0(fileNameRoot,"ThetaDiag") , type="eps" )
# Posterior descriptives:
openGraph(height=3,width=4)
par( mar=c(3.5,0.5,2.5,0.5) , mgp=c(2.25,0.7,0) )
plotPost( codaSamples[,"theta"] , main="theta" , xlab=bquote(theta) )
saveGraph( file=paste0(fileNameRoot,"ThetaPost") , type="eps" )
# Re-plot with different annotations:
plotPost( codaSamples[,"theta"] , main="theta" , xlab=bquote(theta) , 
          cenTend="median" , compVal=0.5 , ROPE=c(0.45,0.55) , credMass=0.90 )
saveGraph( file=paste0(fileNameRoot,"ThetaPost2") , type="eps" )
'''