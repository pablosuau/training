import pandas as pd
import pymc3 as pm
from dbda2e_utilities import diag_mcmc, plot_post

# Load the data (it has to be in the current working directory)
my_data = pd.read_csv('z15n50.csv')
y = my_data.y.values # The y values are in the column named y
n_total = y.shape[0]

# Define the model
# PyMC3 uses a different model by default. Therefore, most of the initialisation and additional parameters
# in the R code are not relevant in the Python version. For instance, we do not need an update (n.iter = 500) 
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
# Posterior descriptives
plot_post(trace['theta'], main = 'theta', xlab = 'theta', cen_tend = 'mode')
# Re-plot with different notation
plot_post(trace['theta'], main = 'theta', xlab = 'theta', cen_tend = 'median', comp_val = 0.5, rope = [0.45, 0.55], cred_mass = 0.9)