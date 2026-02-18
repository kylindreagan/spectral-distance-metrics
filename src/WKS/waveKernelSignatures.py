import numpy as np


def compute_wks(evals, evecs, n_energies=100, sigma=6.0):
    """
    Compute Wave Kernel Signature

    evals  : (k,) eigenvalues
    evecs  : (n,k) eigenvectors
    returns: (n,n_energies) WKS
    """

    evals = np.maximum(evals, 1e-12)

    log_e = np.log(evals)

    e_min, e_max = log_e[1], log_e[-1]

    energies = np.linspace(e_min, e_max, n_energies)

    sigma = (energies[1] - energies[0]) * sigma

    wks = np.zeros((evecs.shape[0], n_energies))

    phi_sq = evecs**2

    for i, e in enumerate(energies):

        weights = np.exp(-(e - log_e)**2 / (2 * sigma**2))

        weights /= weights.sum()  # normalize

        wks[:, i] = phi_sq @ weights

    return wks, np.exp(energies)
