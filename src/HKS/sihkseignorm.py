import numpy as np

def compute_sihks_norm(evals, evecs, n_times=50):
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
    
    return hks_si, times * lambda_1