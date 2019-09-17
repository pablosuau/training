import numpy as np

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