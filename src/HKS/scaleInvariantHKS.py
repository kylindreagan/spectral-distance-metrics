import numpy as np

def compute_scale_invariant_hks(evals, evecs, n_times=50):
    # Ensure non-negative eigenvalues
    evals = np.maximum(evals, 1e-12)
    
    #Normalize by the first non-zero eigenvalue
    lambda_1 = evals[1]  # First non-zero eigenvalue
    evals_normalized = evals / lambda_1
    
    # Automatic time sampling (adjusted for normalized eigenvalues)
    t_min = 4*np.log(10)/evals_normalized[-1]
    t_max = 4*np.log(10)/evals_normalized[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    
    # HKS computation with normalized eigenvalues
    exp_term = np.exp(-np.outer(evals_normalized, times))
    phi_sq = evecs**2
    hks = phi_sq @ exp_term
    
    # Scale-invariant HKS: normalize each row (time scale) by sum
    # This removes the overall scaling factor
    hks_si = hks / np.sum(hks, axis=1, keepdims=True)
    
    return hks_si, times * lambda_1  # Return scaled times for reference


def compute_scale_invariant_hks_logsample(evals, evecs, n_times=50):
    """
    Alt approach using log sampling and normalization
    More robust to scale variations
    """
    # Ensure non-negative eigenvalues
    evals = np.maximum(evals, 1e-12)
    
    # Normalize eigenvalues
    evals_normalized = evals / evals[1]
    
    # Time sampling on a log scale
    t_min = 4*np.log(10)/evals_normalized[-1]
    t_max = 4*np.log(10)/evals_normalized[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    
    # Compute HKS
    exp_term = np.exp(-np.outer(evals_normalized, times))
    phi_sq = evecs**2
    hks = phi_sq @ exp_term
    
    # Apply scale invariance through logarithmic sampling and normalization
    # Normalize each vertex signature
    hks_normalized = hks / np.sum(hks, axis=1, keepdims=True)
    
    # Additionally, normalize by the sum over all vertices at each time
    # This makes the descriptor invariant to global scaling
    hks_si = hks_normalized / np.sum(hks_normalized, axis=0, keepdims=True)
    
    return hks_si, times


def compute_hks_with_scale_estimation(evals, evecs, area=None, n_times=50):
    """
    Compute HKS with scale estimation using surface area if available
    """
    # Ensure non-negative eigenvalues
    evals = np.maximum(evals, 1e-12)
    
    # If area is provided, use it for scale normalization
    if area is not None:
        # Scale eigenvalues by area (for 2D surfaces)
        # For 3D shapes, you might want area^(2/3) or volume^(2/3)
        scale_factor = area
        evals_scaled = evals * scale_factor
    else:
        # Estimate scale from eigenvalues
        # The first eigenvalue scales as 1/area for 2D manifolds
        evals_scaled = evals / evals[1]
    
    # Time sampling
    t_min = 4*np.log(10)/evals_scaled[-1]
    t_max = 4*np.log(10)/evals_scaled[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    
    # Compute HKS
    exp_term = np.exp(-np.outer(evals_scaled, times))
    phi_sq = evecs**2
    hks = phi_sq @ exp_term
    
    # Make scale invariant by normalizing
    hks_si = hks / np.sum(hks, axis=1, keepdims=True)
    
    return hks_si, times