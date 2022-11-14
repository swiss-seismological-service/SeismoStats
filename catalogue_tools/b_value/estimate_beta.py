import numpy as np

def estimate_beta_tinti(magnitudes: list, mc: float, delta_m: float = 0,
    weights = None, axis = None) -> float:
    """ returns the maximum likelyhood b-vlaue
    Source:
        Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
        Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
            America, 77(6), 2125-2134.)

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no 
                    magnitudes belowm mc present)
        mc
        delta_m:    discretization of magnitudes. default is no discretization
    
    Returns:
        beta:       maximum likelihood b-value

    """
    
    if delta_m > 0:
        p = (1 + (delta_m / (np.average(
            magnitudes - mc, weights=weights, axis=axis))))
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average((magnitudes - mc),
            weights=weights, axis=axis)
   
    return beta

def estimate_beta_utsu(magnitudes: list, mc: float, delta_m: float = 0) -> float:
    """ returns the maximum likelyhood b-vlaue
    Source:
        Utsu 1965 (Geophysical bulletin of the Hokkaido University, vol 13, pp
            99-103)

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no 
                    magnitudes belowm mc present)
        mc
        delta_m:    discretization of magnitudes. default is no discretization
    
    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))

    """
    
    beta = 1 / (np.mean(magnitudes) - mc - delta_m/2) 
    
    return beta
