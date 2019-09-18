import numpy as np
import matplotlib.pyplot as plt
from dbda2e_utilities import hdi_of_grid

def bern_grid(theta, p_theta, data, plot_type, show_cent_tend, show_hdi, hdi_mass = 0.95, show_pd = False):
    '''
    Calculate a coin bias (probability of heads) using a Bernoulli likelihood function and a grid approximation
    for the continuous parameter. All the input parameters should be numpy arrays

    Parameters:
        - theta: vector of values between 0 and 1 to indicate the grid positions for the prior of theta
        - p_theta: the probability mass at each point in the grid defined in the previous parameter
          for the prior of theta
        - data: vector of 0's and 1's with the observed data
        - plot_type: type of plot. The accepted values are 'points' or 'bars'
        - show_cent_tend: which metric of central tendency to show. The accepted values are 'mean', 'mode' or 'none'
        - show_hdi: whether to show the HDI or not
        - hdi_mass: probability mass of the HDI. The default value is 0.95.
        _ show_pd: wheter to show the marginal likelihood or not 
    '''
    n_to_plot = min(2001, len(theta))

    # Check for input errors
    if np.logical_or(np.any(theta > 1), np.any(theta < 0)):
        raise ValueError('theta values must be between 0 and 1')
    if np.any(p_theta < 0):
        raise ValueError('p_theta values must be non-negative')
    if np.sum(p_theta) > 1.00001 or np.sum(p_theta) < 1 - 0.00001:
        raise ValueError('p_theta values must sum to 1.0')
    if np.any(np.logical_and(data != 1, data != 0)):
        raise ValueError('data values must be 0 or 1')
    if plot_type != 'bars' and plot_type != 'points':
        raise ValueError('plot type must be either \'points\' or \'bars\'')
    if show_cent_tend not in ['mean', 'mode', 'none']:
        raise ValueError('show cent trend must be \'mean\', \'mode\' or \'none\'')

    # Create a summary of the data
    z = np.sum(data)
    n = len(data.tolist())

    # Compute the Bernoulli likelihood at each value of Theta
    p_data_given_theta = np.power(theta, z) * np.power((1 - theta), n - z)
    # Compute the evidence and the posterior via Bayes' rule:
    p_data = np.sum(p_data_given_theta * p_theta)
    p_theta_given_data = p_data_given_theta * p_theta / p_data

    # Plot the results
    # 1 x 3 panels
    fig, ax = plt.subplots(1, 3)
    fig.set_figwidth(16)
    fig.set_figheight(2)
    # Initialise plot type
    dot_size = 5 # how big to make the plotted dots
    bar_size = 0.01 # how wide to make the bar lines    
    # If the comb has a zillion teeth, it's too many to plot, so plot only a
    # thinned out subset of the teeth.
    n_teeth = len(theta.tolist())
    if n_teeth > n_to_plot:
        thin_idx = np.round(np.linspace(0, n_teeth - 1, n_to_plot)).astype(int)
    else: 
        thin_idx = np.arange(n_teeth)

    # Plot the prior
    if plot_type == 'bars':
        ax[0].bar(theta[thin_idx], p_theta[thin_idx], width = bar_size)
    if plot_type == 'points':
        ax[0].plot(theta[thin_idx], p_theta[thin_idx], 'o', markersize = dot_size)
    ax[0].set_xticks(theta[thin_idx])
    ax[0].set_xlabel('theta')
    ax[0].set_ylabel('p(theta)')
    ax[0].set_title('prior')   
    if show_cent_tend != 'none':
        if show_cent_tend == 'mean':
            mean_theta = np.sum(theta * p_theta)
            if mean_theta > 0.5:
                text_x = 0.05
            else:
                text_x = 0.6
            text = 'mean = {0:.2f}'.format(mean_theta)
        if show_cent_tend == 'mode':
            mode_theta = theta[np.argmax(p_theta)]
            if mode_theta > 0.5:
                text_x = 0.05
            else:
                text_x = 0.6
            text = 'mode = {0:.2f}'.format(mode_theta)
        ax[0].text(text_x, 0.9 * np.max(p_theta), text, fontsize = 14)
    x_ticks = ax[0].get_xticks()
    ax[0].set_xticks(x_ticks[np.round(np.linspace(0, len(x_ticks) - 1, 21)).astype(int)])
    ax[0].tick_params(rotation = 90)


    # Mark the highest density interval. HDI points are not thinned in the plot
    if (show_hdi):
        [indexes, mass, height] = hdi_of_grid(p_theta, hdi_mass)
        text = '{0:.2f}% HDI'.format(100 * mass)
        ax[0].text(np.mean(theta[indexes]), 
                   height, 
                   text, 
                   fontsize = 14, 
                   horizontalalignment = 'center',
                   verticalalignment = 'bottom')
        # Mark the left and right ends of the waterline
        # Find indices at ends of sub-intervals
        in_lim = [indexes[0]] # first point
        for idx in range(1, len(indexes) - 1):
            if indexes[idx] != indexes[idx - 1] + 1 or indexes[idx] != indexes[idx + 1] - 1:
                in_lim.append(indexes[idx]) # include idx
        in_lim.append(indexes[-1]) # last point
        # Mark vertical lines at ends of sub-intervals
        for i in range(len(in_lim)):
            idx = in_lim[i]
            ax[0].plot([theta[idx], theta[idx]], [0, height], 'r--', linewidth = 1.5)
            if i % 2 == 0:
                ax[0].plot([theta[idx], theta[in_lim[i + 1]]], np.repeat(height, 2), 'r--')

    # Plot the likelihood: p(Data|Theta)
    if plot_type == 'bars':
        ax[1].bar(theta[thin_idx], p_data_given_theta[thin_idx], width = bar_size)
    if plot_type == 'points':
        ax[1].plot(theta[thin_idx], p_data_given_theta[thin_idx], 'o', markersize = dot_size)
    ax[1].set_xticks(theta[thin_idx])
    ax[1].set_xlabel('theta')
    ax[1].set_ylabel('p(D|theta)')
    ax[1].set_title('likelihood')   
    if show_cent_tend != 'none':
        if show_cent_tend == 'mean':
            mean_theta = np.sum(theta * p_data_given_theta)
            if mean_theta > 0.5:
                text_x = 0.05
            else:
                text_x = 0.6
            text = 'mean = {0:.2f}'.format(mean_theta)
        if show_cent_tend == 'mode':
            mode_theta = theta[np.argmax(p_data_given_theta)]
            if mode_theta > 0.5:
                text_x = 0.05
            else:
                text_x = 0.6
            text = 'mode = {0:.2f}'.format(mode_theta)
        ax[1].text(text_x, 0.9 * np.max(p_data_given_theta), text, fontsize = 14)
    x_ticks = ax[1].get_xticks()
    ax[1].set_xticks(x_ticks[np.round(np.linspace(0, len(x_ticks) - 1, 21)).astype(int)])
    ax[1].tick_params(rotation = 90)

    # Plot the posterior
    if plot_type == 'bars':
        ax[2].bar(theta[thin_idx], p_theta_given_data[thin_idx], width = bar_size)
    if plot_type == 'points':
        ax[2].plot(theta[thin_idx], p_theta_given_data[thin_idx], 'o', markersize = dot_size)
    ax[2].set_xticks(theta[thin_idx])
    ax[2].set_xlabel('theta')
    ax[2].set_ylabel('p(D|theta)')
    ax[2].set_title('likelihood')   
    if show_cent_tend != 'none':
        if show_cent_tend == 'mean':
            mean_theta = np.sum(theta * p_theta_given_data)
            if mean_theta > 0.5:
                text_x = 0.05
            else:
                text_x = 0.6
            text = 'mean = {0:.2f}'.format(mean_theta)
        if show_cent_tend == 'mode':
            mode_theta = theta[np.argmax(p_theta_given_data)]
            if mode_theta > 0.5:
                text_x = 0.05
            else:
                text_x = 0.6
            text = 'mode = {0:.2f}'.format(mode_theta)
        ax[2].text(text_x, 0.9 * np.max(p_theta_given_data), text, fontsize = 14)
    x_ticks = ax[2].get_xticks()
    ax[2].set_xticks(x_ticks[np.round(np.linspace(0, len(x_ticks) - 1, 21)).astype(int)])
    ax[2].tick_params(rotation = 90)

    # Plot marginal likelihood pData
    if show_pd:
        mean_theta = np.sum(theta * p_theta_given_data)
        if mean_theta > 0.5:
            text_x = 0.05
        else:
            text_x = 0.6
        text = 'p(D) = {0:.2f}'.format(p_data)
        ax[2].text(text_x, 0.8 * np.max(p_theta_given_data), text, fontsize = 14)

    # Mark the highest density interval. HDI points are not thinned in the plot
    if (show_hdi):
        [indexes, mass, height] = hdi_of_grid(p_theta_given_data, hdi_mass)
        text = '{0:.2f}% HDI'.format(100 * mass)
        ax[2].text(np.mean(theta[indexes]), 
                   height, 
                   text, 
                   fontsize = 14, 
                   horizontalalignment = 'center',
                   verticalalignment = 'bottom')
        # Mark the left and right ends of the waterline
        # Find indices at ends of sub-intervals
        in_lim = [indexes[0]] # first point
        for idx in range(1, len(indexes) - 1):
            if indexes[idx] != indexes[idx - 1] + 1 or indexes[idx] != indexes[idx + 1] - 1:
                in_lim.append(indexes[idx]) # include idx
        in_lim.append(indexes[-1]) # last point
        # Mark vertical lines at ends of sub-intervals
        for i in range(len(in_lim)):
            idx = in_lim[i]
            ax[2].plot([theta[idx], theta[idx]], [0, height], 'r--', linewidth = 1.5)
            if i % 2 == 0:
                ax[2].plot([theta[idx], theta[in_lim[i + 1]]], np.repeat(height, 2), 'r--')

    return p_theta_given_data