from src.ShapeDNA.laplaceBeltramiShape import laplace_beltrami_eigenvalues
import unittest
import numpy as np
import trimesh

def create_sphere_mesh(radius=1.0, subdivisions=3):
    """Return vertices and faces separately"""
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    return mesh.vertices, mesh.faces

def scale_vertices(vertices, factor):
    """Scale vertices by factor"""
    return vertices * factor

def rotate_vertices(vertices, angles_deg=None):
    """Rotate vertices by given angles"""
    if angles_deg is None:
        angles_deg = [30, 45, 60]
    
    rx, ry, rz = np.radians(angles_deg)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    rotation_matrix = Rz @ Ry @ Rx
    return vertices @ rotation_matrix.T

class TestShapeDNA(unittest.TestCase):
    
    
    def test_sphere_eigenvalues(self):
        """Test sphere eigenvalues"""
        V, F = create_sphere_mesh(radius=1.0, subdivisions=3) 
        eigenvalues = laplace_beltrami_eigenvalues(V, F, k=20)
        
        self.assertAlmostEqual(eigenvalues[0], 0.0, delta=1e-1)
        
        self.assertTrue(np.all(np.diff(eigenvalues) >= -1e-6))
        
        V_scaled = scale_vertices(V, 2.0)
        eigenvalues_scaled = laplace_beltrami_eigenvalues(V_scaled, F, k=20)
        expected_scaled = eigenvalues / 4.0 
        
        np.testing.assert_allclose(eigenvalues_scaled, expected_scaled, rtol=0.15, atol=1e-2)
    
    def test_isometric_invariance(self):
        """Test eigenvalues are invariant to rotations"""
        V, F = create_sphere_mesh(radius=1.0, subdivisions=3)
        eigenvalues1 = laplace_beltrami_eigenvalues(V, F, k=20)
        V_rotated = rotate_vertices(V, [30, 45, 60])
        eigenvalues2 = laplace_beltrami_eigenvalues(V_rotated, F, k=20)
        np.testing.assert_allclose(eigenvalues1, eigenvalues2, rtol=.01, atol=1e-8)
    
    def test_eigenvalue_ordering(self):
        """Test eigenvalues are sorted ascending"""
        V, F = create_sphere_mesh(radius=1.0)
        eigenvalues = laplace_beltrami_eigenvalues(V, F, k=50)
        # Check sorted ascending
        self.assertTrue(np.all(np.diff(eigenvalues) >= -1e-10))
    
    def test_multiple_eigenvalues_count(self):
        """Test we get requested number of eigenvalues"""
        V, F = create_sphere_mesh(radius=1.0)
        
        for k in [10, 20, 50]:
            eigenvalues = laplace_beltrami_eigenvalues(V, F, k=k)
            # Note: might get fewer if k > n_vertices-1
            self.assertLessEqual(len(eigenvalues), k)
            self.assertGreater(len(eigenvalues), 0)
    
    def test_mass_matrix_types(self):
        """Test different mass matrices give similar eigenvalues"""
        V, F = create_sphere_mesh()
        
        eig_voronoi = laplace_beltrami_eigenvalues(V, F, k=10, mass_matrix_type='voronoi')
        eig_barycentric = laplace_beltrami_eigenvalues(V, F, k=10, mass_matrix_type='barycentric')
        
        # They should be close but not identical
        np.testing.assert_allclose(eig_voronoi, eig_barycentric, rtol=0.5, atol=1e-2)

    def test_negative_eigenvalues(self):
        """Test no negative eigenvalues (except ~0)"""
        V, F = create_sphere_mesh()
        eigenvalues = laplace_beltrami_eigenvalues(V, F, k=50)
        
        # All eigenvalues except the first (which is ~0) should be positive
        self.assertTrue(np.all(eigenvalues[1:] > -1e-10))

    def test_translation_invariance(self):
        """Test eigenvalues invariant to translation"""
        V, F = create_sphere_mesh()
        eigenvalues1 = laplace_beltrami_eigenvalues(V, F, k=20)
        
        V_translated = V + np.array([10, 20, 30])  # Arbitrary translation
        eigenvalues2 = laplace_beltrami_eigenvalues(V_translated, F, k=20)
        
        np.testing.assert_allclose(eigenvalues1, eigenvalues2, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()