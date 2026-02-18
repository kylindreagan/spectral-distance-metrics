from src.ShapeDNA.laplaceBeltramiShape import laplace_beltrami_eigenvalues
from src.HKS.heatKernelSignatures import compute_hks
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
    def compute_mesh_hks(self, mesh, k=100, n_times=20):
        evals, evecs = laplace_beltrami_eigenvalues(mesh, k=k, return_eigenvectors=True)
        evals = np.abs(evals)

        hks, times = compute_hks(evals, evecs, n_times)

        return hks, times
    
    def test_hks_shape(self):
        """HKS matrix has correct dimensions"""
        sphere = create_sphere_mesh()
        hks, times = self.compute_mesh_hks(sphere)

        n_vertices = len(sphere.vertices)

        self.assertEqual(hks.shape[0], n_vertices)
        self.assertEqual(hks.shape[1], len(times))
    
    def test_hks_nonnegative(self):
        sphere = create_sphere_mesh()
        hks, _ = self.compute_mesh_hks(sphere)

        self.assertTrue(np.all(hks >= -1e-10))
    
    def test_rotation_invariance(self):
        cylinder = create_cylinder_mesh()

        hks1, _ = self.compute_mesh_hks(cylinder)

        angle = math.pi / 4
        direction = [1, 0, 0]

        rot_matrix = trimesh.transformations.rotation_matrix(
            angle, direction
        )

        cylinder.apply_transform(rot_matrix)

        hks2, _ = self.compute_mesh_hks(cylinder)

        np.testing.assert_allclose(
            hks1, hks2,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Failed rotation invariance"
        )
    
    def test_translation_invariance(self):
        cylinder = create_cylinder_mesh()

        hks1, _ = self.compute_mesh_hks(cylinder)

        cylinder.apply_translation([10, 20, 30])

        hks2, _ = self.compute_mesh_hks(cylinder)

        np.testing.assert_allclose(
            hks1, hks2,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Failed translation invariance"
        )
    
    def test_scale_invariance_normalized(self):
        sphere = create_sphere_mesh(radius=1.0)
        hks1, _ = self.compute_mesh_hks(sphere)

        sphere_scaled = create_sphere_mesh(radius=2.0)
        hks2, _ = self.compute_mesh_hks(sphere_scaled)

        hks1_n = normalize_hks(hks1)
        hks2_n = normalize_hks(hks2)

        np.testing.assert_allclose(
            hks1_n,
            hks2_n,
            rtol=5e-2,
            atol=5e-2
    )

    
    def test_time_decay(self):
        """HKS should decay as time increases"""

        sphere = create_sphere_mesh()

        hks, _ = self.compute_mesh_hks(sphere)

        # Pick random vertex
        v = 0

        decay = np.diff(hks[v])

        self.assertTrue(
            np.all(decay <= 1e-6),
            msg="HKS is not monotonically decreasing"
        )
    
    def test_heat_trace_consistency(self):
        """Heat trace from eigenvalues matches integrated HKS"""

        sphere = create_sphere_mesh()

        V, F = sphere.vertices, sphere.faces

        evals, evecs = self.compute_mesh_hks(sphere)

        evals = np.abs(evals)

        hks, times = compute_hks(evals, evecs, n_times=20)

        # Vertex areas from mass matrix
        import igl
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        areas = M.diagonal()

        for i, t in enumerate(times):

            heat_trace_spectral = np.sum(np.exp(-evals * t))
            heat_trace_spatial = np.sum(hks[:, i] * areas)

            self.assertAlmostEqual(
                heat_trace_spectral,
                heat_trace_spatial,
                delta=1e-2
            )
    
    def test_noise_robustness(self):
        """HKS robust to small vertex perturbations"""

        sphere = create_sphere_mesh()

        hks_clean, _ = self.compute_mesh_hks(sphere)

        # Add noise
        noisy_vertices = add_noise(sphere.vertices, sigma=0.01)

        noisy_mesh = trimesh.Trimesh(
            vertices=noisy_vertices,
            faces=sphere.faces
        )

        hks_noisy, _ = self.compute_mesh_hks(noisy_mesh)

        # Compare descriptors
        diff = np.linalg.norm(hks_clean - hks_noisy) / np.linalg.norm(hks_clean)

        self.assertLess(
            diff,
            0.1,
            msg="HKS too sensitive to noise"
        )
    
    def test_intra_vs_inter_distance(self):
        """Same shape < Different shape distance"""

        sphere1 = create_sphere_mesh(radius=1.0)
        sphere2 = create_sphere_mesh(radius=1.0)

        cylinder = create_cylinder_mesh()

        hks_s1, _ = self.compute_mesh_hks(sphere1)
        hks_s2, _ = self.compute_mesh_hks(sphere2)
        hks_cyl, _ = self.compute_mesh_hks(cylinder)

        d_same = np.linalg.norm(
            hks_s1.mean(axis=0) - hks_s2.mean(axis=0)
        )

        d_diff = np.linalg.norm(
            hks_s1.mean(axis=0) - hks_cyl.mean(axis=0)
        )

        self.assertLess(d_same, d_diff)

    def test_localization_decay(self):
        """HKS variance decreases over time"""

        sphere = create_sphere_mesh()

        hks, _ = self.compute_mesh_hks(sphere)

        var_start = np.var(hks[:, 0])
        var_end = np.var(hks[:, -1])

        self.assertGreater(
            var_start,
            var_end,
            msg="Heat diffusion not smoothing"
        )





if __name__ == '__main__':
    unittest.main()