#Assistance from https://github.com/raphaelsulzer/ShapeDNA, "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation" Reuter et al 2009
import open3d as o3d
import numpy as np
from scipy.sparse import csr_matrix
import math
from scipy.sparse.linalg import eigsh

def laplace_beltrami_eigenvalues(mesh):
    