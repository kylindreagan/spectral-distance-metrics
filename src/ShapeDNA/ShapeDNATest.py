from src.ShapeDNA.laplaceBeltramiShape import laplace_beltrami_eigenvalues
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

class TestShapeDNA(unittest.TestCase):
    def test_scaled_eigenvalues(self):
        """Test scaled mesh eigenvalues"""
        cylinder = create_cylinder_mesh() 
        eigenvalues = laplace_beltrami_eigenvalues(cylinder, k=50)
        
        self.assertAlmostEqual(eigenvalues[0], 0.0, delta=1e-1)
        
        self.assertTrue(np.all(np.diff(eigenvalues) >= -1e-6))
        
        cylinderScaled = create_cylinder_mesh(radius=2.0, height =4.0) 
        eigenvalues_scaled = laplace_beltrami_eigenvalues(cylinderScaled, k=50)
        expected_scaled = eigenvalues / 4.0 
        
        np.testing.assert_allclose(eigenvalues_scaled, expected_scaled, rtol=1e-5, atol=1e-2, err_msg="Failed scaling test")
    
    def test_isometric_invariance(self):
        """Test eigenvalues are invariant to rotations"""
        cylinder = create_cylinder_mesh()
        eigenvalues1 = laplace_beltrami_eigenvalues(cylinder, k=50)
        angle = math.pi / 4  # 45 degrees in radians
        direction = [1, 0, 0]  # Rotate around the X-axis
        rot_matrix = trimesh.transformations.rotation_matrix(
    angle, direction
)
        cylinder.apply_transform(rot_matrix)
        eigenvalues2 = laplace_beltrami_eigenvalues(cylinder, k=50)
        np.testing.assert_allclose(eigenvalues1[:5], eigenvalues2[:5], rtol=1e-4, atol=1e-10, err_msg=f"Failed small eigenvalues rotation invariance")
        np.testing.assert_allclose(eigenvalues1[5:], eigenvalues2[5:], rtol=0.01, atol=1e-8, err_msg=f"Failed large eigenvalues rotation invariance")
    
    def test_eigenvalue_ordering(self):
        """Test eigenvalues are sorted ascending"""
        sphere = create_sphere_mesh(radius=1.0)
        eigenvalues = laplace_beltrami_eigenvalues(sphere, k=50)
        # Check sorted ascending
        self.assertTrue(np.all(np.diff(eigenvalues) >= -1e-10))
    
    def test_multiple_eigenvalues_count(self):
        """Test we get requested number of eigenvalues"""
        sphere = create_sphere_mesh(radius=1.0)
        
        for k in [10, 20, 50]:
            eigenvalues = laplace_beltrami_eigenvalues(sphere, k=k)
            # Note: might get fewer if k > n_vertices-1
            self.assertLessEqual(len(eigenvalues), k)
            self.assertGreater(len(eigenvalues), 0)
    
    def test_mass_matrix_types(self):
        """Test different mass matrices give similar eigenvalues"""
        sphere = create_sphere_mesh()
        
        eig_voronoi = laplace_beltrami_eigenvalues(sphere, k=20, mass_matrix_type='voronoi')
        eig_barycentric = laplace_beltrami_eigenvalues(sphere, k=20, mass_matrix_type='barycentric')
        
        # They should be close but not identical
        np.testing.assert_allclose(eig_voronoi, eig_barycentric, rtol=1e-5, atol=1e-2, err_msg="Failed mass matric test")

    def test_negative_eigenvalues(self):
        sphere = create_sphere_mesh()
        eigenvalues = laplace_beltrami_eigenvalues(sphere, k=50)
        
        self.assertTrue(np.all(eigenvalues > -1e-10))

    def test_translation_invariance(self):
        """Test eigenvalues invariant to translation"""
        cylinder = create_cylinder_mesh()
        eigenvalues1 = laplace_beltrami_eigenvalues(cylinder, k=50)
        cylinder.apply_translation([10, 20, 30]) 
        eigenvalues2 = laplace_beltrami_eigenvalues(cylinder, k=50)
        np.testing.assert_allclose(eigenvalues1[:5], eigenvalues2[:5], rtol=1e-4, atol=1e-10, err_msg=f"Failed small eigenvalues rotation invariance")
        np.testing.assert_allclose(eigenvalues1[5:], eigenvalues2[5:], rtol=0.01, atol=1e-8, err_msg=f"Failed large eigenvalues rotation invariance")

if __name__ == '__main__':
    unittest.main()