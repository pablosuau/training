
import numpy as np
from math import ceil
from scipy.stats import beta
import matplotlib.pyplot as plt
from dbda2e_utilities import plot_post, effective_size

# This is a modified version of the original R file in which I am paametrising some
# of the elements of the original file

def bern_metrop(proposal_sd):
  # Specify the data to be used in the likelihood function
  my_data = np.concatenate((np.repeat(0, 6), np.repeat(1, 14)), axis = None) 

  # Define the Bernoulli likelihood function p(D|theta)
  # The argument theta could be a numpy array, not just a scalar
  def likelihood(theta, data):
    z = np.sum(data)
    n = data.shape[0]

    if type(theta) != np.ndarray:
      theta = np.array([theta])

    p_data_given_theta = theta ** z * (1 - theta) ** (n - z)
    # The theta values passed into this function are generated at random,
    # and therefore might be inadvertently greater than 1 or less than 0.
    # The likelihood for theta > 1 or for theta < 0 is zero:
    p_data_given_theta[np.where(np.logical_or(theta > 1, theta < 0))] = 0
    
    return p_data_given_theta

  # Define the prior density function
  def prior(theta):
    if type(theta) != np.ndarray:
      theta = np.array([theta])

    p_theta = beta.pdf(theta, 1, 1)
    # The theta values passed into this function are generated at random,
    # and therefore might be inadvertently greater than 1 or less than 0.
    # The prior for theta > 1 or for theta < 0 is zero:
    p_theta[np.where(np.logical_or(theta > 1, theta < 0))] = 0

    return p_theta

  # Define the relative probability of the target distribution, 
  # as a function of vector theta. For our application, this
  # target distribution is the unnormalized posterior distribution.
  def target_rel_prob(theta , data):
    target_rel_prob =  likelihood(theta , data) * prior(theta)

    return target_rel_prob

  # Specify the length of the trajectory, i.e., the number of jumps to try:
  traj_length = 50000 # arbitrary large number
  # Initialize the vector that will store the results:
  trajectory = np.zeros(traj_length)
  # Specify where to start the trajectory:
  trajectory[0] = 0.01 # arbitrary value
  # Specify the burn-in period:
  burn_in = ceil(0.0 * traj_length) # arbitrary number, less than trajLength
  # Initialize accepted, rejected counters, just to monitor performance:
  n_accepted = 0
  n_rejected = 0

  # Now generate the random walk. The 't' index is time or trial in the walk.
  # Specify seed to reproduce same random walk:
  np.random.seed(47405)
  # Specify standard deviation of proposal distribution:
  for t in range(traj_length - 1):
    current_position = trajectory[t]
    # Use the proposal distribution to generate a proposed jump.
    proposed_jump = np.random.normal(loc = 0, scale = proposal_sd )
    # Compute the probability of accepting the proposed jump.
    prob_accept = min(1,
                      target_rel_prob(current_position + proposed_jump, my_data) / \
                      target_rel_prob(current_position, my_data))
    # Generate a random uniform value from the interval [0,1] to
    # decide whether or not to accept the proposed jump.
    if np.random.uniform() < prob_accept:
      # accept the proposed jump
      trajectory[t + 1] = current_position + proposed_jump
      # increment the accepted counter, just to monitor performance
      if (t > burn_in):
        n_accepted = n_accepted + 1
    else:
      # reject the proposed jump, stay at current position
      trajectory[t + 1] = current_position
      # increment the rejected counter, just to monitor performance
      if t > burn_in:
        n_rejected = n_rejected + 1

  # Extract the post-burnIn portion of the trajectory.
  accepted_traj = trajectory[burn_in:]

  # End of Metropolis algorithm.

  #-----------------------------------------------------------------------
  # Display the chain.


  fig, ax = plt.subplots(3, 1)

  # Posterior histogram
  param_info = plot_post(accepted_traj, 
                         xlim = [0, 1], 
                         xlab = 'theta', 
                         main = 'Prpsl.SD ' + \
                                str(proposal_sd) + \
                                '\nEff.Sz. ' + \
                                str(round(effective_size(accepted_traj), 1)),
                         ax = ax[0])

  # Trajectory, a.k.a. trace plot, end of chain:
  idx_to_plot = range((traj_length - 100), traj_length)
  ax[1].plot(trajectory[idx_to_plot],
             idx_to_plot,
             '-o',
             color = 'C0')
  ax[1].set_title('end of chain')
  ax[1].set_xlabel('theta')
  ax[1].set_ylabel('step in chain')
  # Display proposal SD and acceptance ratio in the pot
  ax[1].text(ax[1].get_xlim()[0] + 0.01, 
             traj_length - 10,
             'N[acc] / N[pro] = ' + str(round(n_accepted / len(accepted_traj), 3)))
  # Trajectory, a.k.a. trace plot, beginning of chain:
  idx_to_plot = range(0, 100)
  ax[2].plot(trajectory[idx_to_plot],
             idx_to_plot,
             '-o',
             color = 'C0')
  ax[2].set_title('beginning of chain')
  ax[2].set_xlabel('theta')
  ax[2].set_ylabel('step in chain')
  # Indicate burn in limit (might not be visible if not in range)
  if burn_in > 0 and burn_in < max(idx_to_plot):
    ax[2].axhline(y = burn_in, ls =':', color = 'black')
    ax[2].text(ax[2].get_xlim()[0] + 0.01, 
               burn_in + 2,
               'burn in')
  fig.set_figheight(12)
  fig.set_figwidth(5)
  plt.tight_layout()