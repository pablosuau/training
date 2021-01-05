import pandas as pd
import numpy as np
from jags_ydich_xnomssubj_mbinombetaomegakappa import gen_mcmc, smry_mcmc, plot_mcmc
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
  parameter_names = [v for v in mcmc_trace.varnames if 'log' not in v] 
  for par_name in list(np.array(parameter_names)[[0, 1, 2, len(parameter_names) - 1]]):
      diag_mcmc(mcmc_trace, par_name = par_name, range_0_1 = 'theta' in par_name or 'omega' in par_name)

  # Get summary statistics of chain
  summary_info = smry_mcmc(mcmc_trace, comp_val = 0.5, comp_val_diff = 0.0, diff_id_vec = [0, 13, 27])

  # Display posterior information
  plot_mcmc(mcmc_trace, my_data, comp_val = 0.5, diff_id_vec = [0, 13, 27], comp_val_diff = 0)