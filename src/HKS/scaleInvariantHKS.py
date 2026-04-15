import numpy as np

def compute_scale_invariant_hks(evals, evecs, n_times=100, n_freqs=16):
    evals = np.maximum(evals, 1e-12)

    # Log-uniform time sampling (scale-invariant range)
    t_min = 4 * np.log(10) / evals[-1]
    t_max = 4 * np.log(10) / evals[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)

    # HKS matrix (n_vertices, n_times)
    exp_term = np.exp(-np.outer(evals, times))
    hks = (evecs ** 2) @ exp_term

    # Real FFT, magnitude, skip DC, normalise
    spec = np.abs(np.fft.rfft(hks, axis=1))
    dc = spec[:, 0:1] + 1e-12
    sihks = spec[:, 1:1+n_freqs] / dc

    return sihks, times