from src.ShapeDNA.laplaceBeltramiShape import laplace_beltrami_eigenvalues
from src.WKS.waveKernelSignatures import compute_wks
import unittest
import numpy as np
import trimesh
import math

def create_sphere_mesh(radius=1.0, subdivisions=5):
    return trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)

def create_cylinder_mesh(radius=1.0, height=2.0, subdivisions=20):
    return trimesh.creation.cylinder(radius=radius, height=height, subdivisions=subdivisions)

def scale_vertices(vertices, factor):
    return vertices * factor

def normalize_hks(hks):
    trace = np.sum(hks, axis=0, keepdims=True)
    return hks / trace

def add_noise(vertices, sigma=0.01):
    noise = np.random.normal(scale=sigma, size=vertices.shape)
    return vertices + noise

class TestHKS(unittest.TestCase):
    def test_wks_nonnegative(self):
        sphere = create_sphere_mesh()

        wks, _ = self.compute_mesh_wks(sphere)

        self.assertTrue(np.all(wks >= -1e-10))
    
    def test_wks_rotation_invariance(self):
        cyl = create_cylinder_mesh()

        wks1, _ = self.compute_mesh_wks(cyl)

        rot = trimesh.transformations.rotation_matrix(
            np.pi/3, [0,1,0]
        )

        cyl.apply_transform(rot)

        wks2, _ = self.compute_mesh_wks(cyl)

        np.testing.assert_allclose(wks1, wks2, rtol=1e-4)
    
    def test_wks_noise_robustness(self):
        #WKS is slightly less robust than HKS but still stable.
        sphere = create_sphere_mesh()

        wks_clean, _ = self.compute_mesh_wks(sphere)

        noisy_vertices = add_noise(sphere.vertices, 0.01)

        noisy_mesh = trimesh.Trimesh(
            vertices=noisy_vertices,
            faces=sphere.faces
        )

        wks_noisy, _ = self.compute_mesh_wks(noisy_mesh)

        diff = np.linalg.norm(wks_clean - wks_noisy) / np.linalg.norm(wks_clean)

        self.assertLess(diff, 0.15)


