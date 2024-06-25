# =============================================================================
# PACKAGES
# =============================================================================
import pandas as pd
from collections import Counter
import scipy.integrate as it
import numpy as np
import time 

# =============================================================================
# FUNCTIONS
# =============================================================================

def to_array(data, task_name, add_noise = True, verbose = False):
    """
    Transform dataframe pandas to array.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
    task_name : str
        task name.
    add_noise : Bool, optional
        If add noise. The default is True.
    verbose : Bool, optional
        The default is False.

    Returns
    -------
    estimate_list : list
    """
    # select task
    data = data.loc[
        data[task_name] > 0,
        task_name
        ]
    # transform in list
    estimate_list = [
        duration
        for duration in data.index
        for i in range(int(data[duration]))
        ]
    if add_noise: 
        estimate_list = add_noise(estimate_list)
    estimate_list.sort()
    return estimate_list

def loss_density(observed, estimate, verbose = False):
    """
    Loss function

    Parameters
    ----------
    observed : value
        Observed value.
    estimate : list
        Estimate values.
    verbose : Bool, optional

    Returns
    -------
    error : float
        loss metric.
    """
    if verbose: start_time = time.time()
    # get Z freqs and percentages
    n = len(estimate)
    freq_z = Counter(estimate)
    p_z = {z: freq/n for z, freq in freq_z.items()}
    #
    #print(p_z)
    L =  - 2*p_z.get(observed, 0)
    for z, p in p_z.items():
        L += (p**2)/len(p_z)
    if verbose:
        (
            pd.concat([
                pd.DataFrame({'duration': estimate}).assign(label='estimate')
            ])
            .reset_index(drop=True)
            .pipe(lambda df: (
                sns.displot(data=df, x='duration', hue='label', fill=True, kind='hist', stat='probability')
            ))
        )
    return L


def pinball_loss(observed, estimate, alpha):
    """
    Calculate Pinball Loss

    Parameters
    ----------
    observed : value
        Observed value.
    estimate : value
        value values.
    alpha: float
        alpha quantile
        
    Returns
    -------
    result : float
    """
    if observed >= estimate: 
        result = (observed-estimate)*alpha
    else:
        result = (estimate-observed)*(1-alpha)
    return result



def calculate_quantiles(observed, estimate, num = 1000):
    x_alpha = np.linspace(0, 1, num)
    L_alpha = [
        pinball_loss(
            observed = observed, 
            estimate = np.quantile(estimate, q = alpha), 
            alpha=alpha
            ) 
        for alpha in x_alpha]
    return x_alpha, L_alpha

def weighted_pinball_loss(
        observed, estimate, 
        weighting = "right-tail", num = 1000, verbose = False
        ):
    """
    Loss function

    Parameters
    ----------
    observed : value
        Observed value.
    estimate : list
        Estimate values.
    weighting : string
        weighting function format 
    num : int
        Number of axis for itegrate
    verbose : Bool, optional

    Returns
    -------
    L_w : float
        loss metric.
    """
    x_alpha, L_alpha = calculate_quantiles(
        observed=observed, estimate=estimate, num=num)

    if weighting == "two-tailed":
        weighting_function = (x_alpha-.5)**2
    elif weighting == "left-tail":
        weighting_function = (1-x_alpha)**2
    elif weighting == "right-tail":
        weighting_function = (x_alpha)**2
    elif weighting == "centered":
        weighting_function = 0.25 - (x_alpha-.5)**2
    elif weighting == None:
        weighting_function = 1
    
    L_w = it.simps(weighting_function*L_alpha, x_alpha)
    return L_w


def loss(observed, estimate, loss_type = 'density'):
    """
    Apply loss function by loss_type.

    Parameters
    ----------
    observed : value
        Observed value.
    estimate : list
        Estimate values.
    verbose : Bool, optional
    loss_type : string, optional
        Loos format. The default is 'density'.

    Returns
    -------
    loss : float
        loss metric.
    """
    if loss_type == 'density':
        return loss_density(observed, estimate, verbose = False)
    elif loss_type == "two-tailed":
        return weighted_pinball_loss(observed, estimate, weighting = "two-tailed", verbose = False)
    elif loss_type == "left-tail":
        return weighted_pinball_loss(observed, estimate, weighting = "left-tail",  verbose = False)
    elif loss_type == "right-tail":
        return weighted_pinball_loss(observed, estimate, weighting = "right-tail", verbose = False)
    elif loss_type == "centered":
        return weighted_pinball_loss(observed, estimate, weighting = "centered", verbose = False)
    elif loss_type == "pinball":
        return weighted_pinball_loss(observed, estimate, weighting = None, verbose = False)
    elif loss_type == "mse":
        return (observed-np.mean(estimate))**2


def calculate_metrics(observed, estimate):
    metrics = {
        'density': loss_density(observed, estimate, verbose = False),
        'mse': (observed-np.mean(estimate))**2
        }
    # opt calculate pinball losses
    x_alpha, L_alpha = calculate_quantiles(
        observed=observed, estimate=estimate)

    for weighting in ["two-tailed", "left-tail", "right-tail", "centered", "pinball"]:
        if weighting == "two-tailed":
            weighting_function = (x_alpha-.5)**2
        elif weighting == "left-tail":
            weighting_function = (1-x_alpha)**2
        elif weighting == "right-tail":
            weighting_function = (x_alpha)**2
        elif weighting == "centered":
            weighting_function = 0.25 - (x_alpha-.5)**2
        elif weighting == "pinball":
            weighting_function = 1
        metrics[weighting] = it.simps(weighting_function*L_alpha, x_alpha)

    return metrics