import numpy as np
from scipy.stats import beta
from scipy.special import betaln
import matplotlib.pyplot as plt
from dbda2e_utilities import hdi_of_icdf

def bern_beta(prior_beta_ab, data, plot_type, show_cent_tend, show_hdi, hdi_mass = 0.95, show_pd = False):
    '''
    Updates the shape parameters of a Beta distribution (posterior) based on a prior beta and a 
    set of observations - coin tosses. The calculation was derived analytically by taking advantage
    of conjugate priors. 

    Parameters:
        - prior_beta_ab: a list containing shape parameters of the prior beta distribution
        - data: an array of 0's and 1's with the observed data
        - plot_type: type of plot. The accepted values are 'points' or 'bars'
        - show_cent_tend: which metric of central tendency to show. The accepted values are 'mean', 'mode' or 'none'
        - show_hdi: whether to show the HDI or not
        - hdi_mass: probability mass of the HDI. The default value is 0.95.
        _ show_pd: wheter to show the marginal likelihood or not 
    '''
    if np.any(np.array(prior_beta_ab) < 0):
        raise ValueError('prior beta ab values must be nonnegative')
    if len(prior_beta_ab) != 2:
        raise ValueError('prior beta ab must be a vector of two values')
    if np.logical_not(np.all(np.logical_or(data == 1, data == 0))):
        raise ValueError('data vaues must be 0 or 1')
    if plot_type != 'bars' and plot_type != 'points':
        raise ValueError('plot type must be either \'points\' or \'bars\'')
    if show_cent_tend not in ['mean', 'mode', 'none']:
        raise ValueError('show cent trend must be \'mean\', \'mode\' or \'none\'')

    # For notational convenience rename components of prior_beta_ab
    a = prior_beta_ab[0]
    b = prior_beta_ab[1]

    # Create summary values of data
    z = np.sum(data) # Number of 1's in data
    n = len(data.tolist())

    theta = np.arange(0.001, 0.999, 0.001) # Points for plotting
    p_theta = beta.pdf(theta, a, b) # Prior for plotting
    p_theta_given_data = beta.pdf(theta, a + z, b + n - z) # Posterior for plotting
    p_data_given_theta = np.power(theta, z) * np.power(1 - theta, n - z) # likelihood for plotting

    # Compute the evidence for optional display
    # Using data transformation to preven underflow errors for large a, b values
    p_data = np.exp(betaln(z + 1, n - z + b) - betaln(a, b))

    # Plot the results
    # 1 x 3 panels
    fig, ax = plt.subplots(1, 3)
    fig.set_figwidth(16)
    fig.set_figheight(2)
    # Initialise plot type
    dot_size = 5 # how big to make the plotted dots
    bar_size = 0.01 # how wide to make the bar lines   
    # y limits for prior and posterior:
    y_lim = [0, 1.1 * max(np.max(p_theta), np.max(p_theta_given_data))]

    # Plot the prior
    if plot_type == 'bars':
        ax[0].bar(theta, p_theta, width = bar_size)
    if plot_type == 'points':
        ax[0].plot(theta, p_theta, 'o', markersize = dot_size)
    ax[0].set_xticks(theta)
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
    ax[0].set_ylim(y_lim)

    # Mark the highest density interval. HDI points are not thinned in the plot
    if (show_hdi):
        if (a + b - 2 > 0):
            hdi_info = hdi_of_icdf(beta, cred_mass = hdi_mass, a = a, b = b)
            hdi_height = np.mean([beta.pdf(hdi_info[0], a, b), 
                                  beta.pdf(hdi_info[1], a, b)])
            text = '{0:.2f}% HDI'.format(100 * hdi_mass)
            ax[0].text(np.mean(hdi_info), 
                       hdi_height, 
                       text, 
                       fontsize = 14, 
                       horizontalalignment = 'center',
                       verticalalignment = 'bottom')
            # Mark the left and right ends of the waterline
            ax[0].plot(hdi_info, np.repeat(hdi_height, 2), 'r--')
            ax[0].plot(np.repeat(hdi_info[0], 2), [0, hdi_height], 'r--', linewidth = 1.5)
            ax[0].plot(np.repeat(hdi_info[1], 2), [0, hdi_height], 'r--', linewidth = 1.5)

  ## Mark the ROPE
  #if ( !is.null(ROPE) ) {
  #  #pInRope = ( pbeta( ROPE[2] , shape1=a+z , shape2=b+N-z ) 
  #  #            - pbeta( ROPE[1] , shape1=a+z , shape2=b+N-z ) )
  #  pInRope = ( pbeta( ROPE[2] , shape1=a , shape2=b ) 
  #              - pbeta( ROPE[1] , shape1=a , shape2=b ) )
  #  ropeTextHt = 0.7*yLim[2]
  #  ropeCol = "darkred"
  #  lines( c(ROPE[1],ROPE[1]) , c(-0.5,ropeTextHt) , type="l" , lty=2 , 
  #         lwd=1.5 , col=ropeCol )
  #  lines( c(ROPE[2],ROPE[2]) , c(-0.5,ropeTextHt) , type="l" , lty=2 , 
  #         lwd=1.5 , col=ropeCol )
  #  text( mean(ROPE) , ropeTextHt ,
  #        paste0(ROPE[1],"<",round(pInRope,4)*100,"%<",ROPE[2]) ,
  #        adj=c(0.5,-0.15) , cex=1.2 , col=ropeCol )    
  #}
  
    # Plot the likelihood: p(Data|Theta)
    if plot_type == 'bars':
        ax[1].bar(theta, p_data_given_theta, width = bar_size)
    if plot_type == 'points':
        ax[1].plot(theta, p_data_given_theta, 'o', markersize = dot_size)
    ax[1].set_xticks(theta)
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
        ax[2].bar(theta, p_theta_given_data, width = bar_size)
    if plot_type == 'points':
        ax[2].plot(theta, p_theta_given_data, 'o', markersize = dot_size)
    ax[2].set_xticks(theta)
    ax[2].set_xlabel('theta')
    ax[2].set_ylabel('p(theta|D)')
    ax[2].set_title('posterior')  
    ax[2].set_ylim(y_lim) 
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
        if (a + b + n - 2 > 0):
            hdi_info = hdi_of_icdf(beta, cred_mass = hdi_mass, a = a + z, b = b + n - z)
            hdi_height = np.mean([beta.pdf(hdi_info[0], a + z, b + n - z), 
                                  beta.pdf(hdi_info[1], a + z, b + n - z)])
            text = '{0:.2f}% HDI'.format(100 * hdi_mass)
            ax[2].text(np.mean(hdi_info), 
                       hdi_height, 
                       text, 
                       fontsize = 14, 
                       horizontalalignment = 'center',
                       verticalalignment = 'bottom')
            # Mark the left and right ends of the waterline
            ax[2].plot(hdi_info, np.repeat(hdi_height, 2), 'r--')
            ax[2].plot(np.repeat(hdi_info[0], 2), [0, hdi_height], 'r--', linewidth = 1.5)
            ax[2].plot(np.repeat(hdi_info[1], 2), [0, hdi_height], 'r--', linewidth = 1.5)

  ## Mark the ROPE
  #if ( !is.null(ROPE) ) {
  #  pInRope = ( pbeta( ROPE[2] , shape1=a+z , shape2=b+N-z ) 
  #              - pbeta( ROPE[1] , shape1=a+z , shape2=b+N-z ) )
  #  ropeTextHt = 0.7*yLim[2]
  #  ropeCol = "darkred"
  #  lines( c(ROPE[1],ROPE[1]) , c(-0.5,ropeTextHt) , type="l" , lty=2 , 
  #         lwd=1.5 , col=ropeCol )
  #  lines( c(ROPE[2],ROPE[2]) , c(-0.5,ropeTextHt) , type="l" , lty=2 , 
  #         lwd=1.5 , col=ropeCol )
  #  text( mean(ROPE) , ropeTextHt ,
  #        paste0(ROPE[1],"<",round(pInRope,4)*100,"%<",ROPE[2]) ,
  #        adj=c(0.5,-0.15) , cex=1.2 , col=ropeCol )    
  #}
  
    return [a + z, b + n - z]