import pandas as pd
from jags_ydich_xnomssubj_mbernbeta import gen_mcmc, smry_mcmc, plot_mcmc
from dbda2e_utilities import diag_mcmc

# Example for jags_ydich_xnomssubj_mbernbeta.py

def jags_ydich_xnomssubj_mbernbeta_example(data_csv = 'z6n8z2n7.csv'):
	my_data = pd.read_csv(data_csv)
	
	# Generate the MCMC chain
	mcmc_trace = gen_mcmc(data = my_data, num_saved_steps = 50000)

	# Display diagnostics of chain. for specificed parameters
	parameter_names = [v for v in mcmc_trace.varnames if 'logodds' not in v] # get all parameter names
	for par_name in parameter_names:
		diag_mcmc(mcmc_trace, par_name = par_name)

	# Get summary statistics of chain
	summary_info = smry_mcmc(mcmc_trace, comp_val = None, comp_val_diff = 0.0)

	# Display posterior information
	plot_mcmc(mcmc_trace, data = my_data, comp_val = None, comp_val_diff = 0.0)