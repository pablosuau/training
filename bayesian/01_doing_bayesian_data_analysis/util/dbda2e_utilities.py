import numpy as np
from scipy.optimize import fmin
from scipy.stats import *

def hdi_of_icdf(icdf_name, cred_mass = 0.95, tol = 1e-8, **args):
    '''
    Calculate HDI from an inverse cumulative density function.
    Implementation from https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region

    Parameters:
        - icdf_name: the ICDF Python function
        - cred_mass: the desired mass of the HDI region
        - tol: tol is passed to the optimize's fmin function
        - args: named parameters related to icdf_name
    Returns: 
        - A list with the HDI limits
    '''
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incred_mass =  1.0 - cred_mass

    def interval_width(low_tail_pr):
        return distri.ppf(cred_mass + low_tail_pr) - distri.ppf(low_tail_pr)

    # find low_tail_pr that minimizes interval_width
    hdi_low_tail_pr = fmin(interval_width, incred_mass, ftol = tol, disp = False)[0]
    # return interval as array([low, high])
    return distri.ppf([hdi_low_tail_pr, cred_mass + hdi_low_tail_pr]).tolist()

def hdi_of_grid(prob_mass_vector, cred_mass = 0.95):
    '''
    Appoximate HDI estimation from a grid

    Parameters:
        - prob_mass_vector: array of probability masses at each grid point
        - cred_mass: the desired mass of the HDI region
    Returns:
        - A list with: a list of the indices in the HDI, the total mass of the
          included indices, the smallest component probability mass in the HDI
    '''
    sorted_prob_mass = np.argsort(-prob_mass_vector)
    hdi_height_idx = np.where(np.cumsum(prob_mass_vector[sorted_prob_mass]) >= cred_mass)[0][0]
    hdi_height = prob_mass_vector[sorted_prob_mass[hdi_height_idx]]
    hdi_indexes = np.where(prob_mass_vector >= hdi_height)[0]
    hdi_mass = np.sum(prob_mass_vector[hdi_indexes])
    return [hdi_indexes, hdi_mass, hdi_height]