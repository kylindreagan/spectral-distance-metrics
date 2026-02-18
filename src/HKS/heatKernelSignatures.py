import numpy as np

def compute_hks(evals, evecs, n_times=50):
    # Ensure non-negative eigenvalues
    evals = np.maximum(evals, 1e-12)

    # Automatic time sampling
    t_min = 4*np.log(10)/evals[-1]
    t_max = 4*np.log(10)/evals[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)

    # HKS computation
    exp_term = np.exp(-np.outer(evals, times))
    phi_sq = evecs**2
    hks = phi_sq @ exp_term

    return hks, times