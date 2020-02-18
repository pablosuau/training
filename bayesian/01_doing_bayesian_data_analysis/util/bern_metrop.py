
import numpy as np
from math import ceil
from scipy.stats import beta
import matplotlib.pyplot as plt
from dbda2e_utilities import plot_post, effective_size

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
proposal_sd = [0.02,0.2,2.0][2]
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
plt.tight_layout()

# Posterior histogram
param_info = plot_post(accepted_traj, 
                       xlim = [0, 1], 
                       xlab = 'theta', 
                       main = 'Prpsl.SD ' + \
                              str(proposal_sd) + \
                              '\nEff.Sz. ' + \
                              str(round(effective_size(accepted_traj), 1)),
                       ax = ax[0])

'''
graphics.off()
rm(list=ls(all=TRUE))
fileNameRoot="BernMetrop" # for output filenames
source("DBDA2E-utilities.R")





#-----------------------------------------------------------------------
# Display the chain.

openGraph(width=4,height=8)
layout( matrix(1:3,nrow=3) )
par(mar=c(3,4,2,1),mgp=c(2,0.7,0))


# Trajectory, a.k.a. trace plot, end of chain:
idxToPlot = (trajLength-100):trajLength
plot( trajectory[idxToPlot] , idxToPlot , main="End of Chain" ,
      xlab=bquote(theta) , xlim=c(0,1) , ylab="Step in Chain" ,
      type="o" , pch=20 , col="skyblue" , cex.lab=1.5 )
# Display proposal SD and acceptance ratio in the plot.
text( 0.0 , trajLength , adj=c(0.0,1.1) , cex=1.75 ,
      labels = bquote( frac(N[acc],N[pro]) == 
                       .(signif( nAccepted/length(acceptedTraj) , 3 ))))

# Trajectory, a.k.a. trace plot, beginning of chain:
idxToPlot = 1:100
plot( trajectory[idxToPlot] , idxToPlot , main="Beginning of Chain" ,
      xlab=bquote(theta) , xlim=c(0,1) , ylab="Step in Chain" ,
      type="o" , pch=20 , col="skyblue" , cex.lab=1.5 )
# Indicate burn in limit (might not be visible if not in range):
if ( burnIn > 0 ) {
  abline(h=burnIn,lty="dotted")
  text( 0.5 , burnIn+1 , "Burn In" , adj=c(0.5,1.1) )
}

#saveGraph( file=paste0( fileNameRoot , 
#                        "SD" , proposalSD ,
#                        "Init" , trajectory[1] ) , type="eps" )

#------------------------------------------------------------------------
© 2020 GitHub, Inc.
'''