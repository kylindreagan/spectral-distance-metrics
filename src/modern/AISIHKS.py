#Average Increment SIHKS
import numpy as np

def compute_aisihks(sihks):
    diff = np.diff(sihks, axis=1)
    return diff