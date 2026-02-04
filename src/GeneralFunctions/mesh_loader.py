from GeneralFunctions.shape_reader import read_mesh_file
import os
import numpy as np
import igl

class MeshLoader:
    """Advanced mesh loader with filtering and statistics"""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.meshes = {}
        self.non_manifold_meshes = []
        self.stats = {}
        
    def load(self, 
             min_vertices=10,
             max_vertices=1000000,
             require_manifold=False,
             manifold_type='edge',
             fix_nonmanifold=False,
             extensions=['.off', '.ply']):
        
        self.meshes = {}
        self.non_manifold_meshes = []
        failed_files = []

        mesh_files = self._find_mesh_files(extensions)

        print(f"Found {len(mesh_files)} mesh files")
        
        for filepath in mesh_files:
            try:
                V, F, name = read_mesh_file(filepath)
                
                if V is None or F is None:
                    failed_files.append(filepath)
                    continue
                
                # Apply filters
                if V.shape[0] < min_vertices:
                    continue
                if V.shape[0] > max_vertices:
                    continue

                manifold_info = self._check_manifold(F, manifold_type)

                if require_manifold and not manifold_info['is_manifold']:
                        if fix_nonmanifold:
                            V_fixed, F_fixed = self._fix_nonmanifold_mesh(V, F)
                            if V_fixed is not None:
                                V, F = V_fixed, F_fixed
                                print(f"  Fixed non-manifold mesh")
                                # Re-check after fixing
                                manifold_info = self._check_manifold(F, manifold_type)
                            else:
                                self.non_manifold_meshes.append({
                                    'name': name,
                                    'filepath': filepath,
                                    'reason': manifold_info.get('non_manifold_reason', 'unknown'),
                                    'info': manifold_info
                                })
                                print(f"  Skipped: Non-manifold (could not fix)")
                                continue
                        else:
                            self.non_manifold_meshes.append({
                                'name': name,
                                'filepath': filepath,
                                'reason': manifold_info.get('non_manifold_reason', 'unknown'),
                                'info': manifold_info
                            })
                            print(f"  Skipped: Non-manifold")
                            continue
                
                # Store mesh
                self.meshes[name] = {
                    'V': V,
                    'F': F,
                    'filepath': filepath,
                    'vertices': V.shape[0],
                    'faces': F.shape[0],
                    'bounds': self._compute_bounds(V),
                    'manifold': manifold_info,
                    'watertight': self._is_watertight(F)
                }

                print(f"  Loaded: {V.shape[0]} vertices, {F.shape[0]} faces, "
                      f"Manifold: {manifold_info['is_manifold']}")
            
            except Exception as e:
                failed_files.append((filepath, str(e)))
                print(f"  Error: {e}")
            
            self._compute_statistics()
        
            return self.meshes, failed_files, self.non_manifold_meshes
    
    def _find_mesh_files(self, extensions):
        """Find all mesh files with given extensions"""
        mesh_files = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    mesh_files.append(os.path.join(root, file))
        return mesh_files
    
    def _compute_bounds(self, V):
        """Compute bounding box"""
        return {
            'min': V.min(axis=0),
            'max': V.max(axis=0),
            'center': V.mean(axis=0),
            'size': V.max(axis=0) - V.min(axis=0)
        }
    
    def _check_manifold(self, F, manifold_type='edge'):
        info = {
            'is_manifold': False,
            'edge_manifold': False,
            'vertex_manifold': False,
            'non_manifold_reason': None,
            'non_manifold_edges': 0,
            'boundary_edges': 0,
            'is_watertight': False
        }
        
        try:
            info['edge_manifold'] = igl.is_edge_manifold(F)

            B = igl.boundary_loop(F)
            info['boundary_edges'] = len(B)
            info['is_watertight'] = (len(B) == 0)

            E = igl.edges(F)
            E_sorted = np.sort(E, axis=1) # Sort edges to make them undirected
            unique_edges, counts = np.unique(E_sorted, axis=0, return_counts=True)
            info['non_manifold_edges'] = np.sum(counts > 2)

            if manifold_type == 'edge':
                info['is_manifold'] = info['edge_manifold']
                if not info['edge_manifold']:
                    info['non_manifold_reason'] = 'non-edge-manifold'
            elif manifold_type == 'vertex':
                info['vertex_manifold'] = self._check_vertex_manifold(F)
                info['is_manifold'] = info['vertex_manifold']
                if not info['vertex_manifold']:
                    info['non_manifold_reason'] = 'non-vertex-manifold'
            elif manifold_type == 'both':
                info['vertex_manifold'] = self._check_vertex_manifold(F)
                info['is_manifold'] = info['edge_manifold'] and info['vertex_manifold']
                if not info['edge_manifold']:
                    info['non_manifold_reason'] = 'non-edge-manifold'
                elif not info['vertex_manifold']:
                    info['non_manifold_reason'] = 'non-vertex-manifold'
            
            if info['non_manifold_edges'] > 0:
                info['is_manifold'] = False
                info['non_manifold_reason'] = f'has {info["non_manifold_edges"]} non-manifold edges'
        except Exception as e:
            info['error'] = str(e)
            info['non_manifold_reason'] = f'check failed: {e}'
        return info
    
    def _check_vertex_manifold(self, F):
        try:
            B = igl.boundary_loop(F)
            V = F.max() + 1  # Approximate vertex count from faces
            E = igl.edges(F).shape[0]
            euler_char = V - E + F.shape[0]
            return igl.is_edge_manifold(F) and abs(euler_char) <= 2 * V
        except:
            return igl.is_edge_manifold(F)
    
    def _fix_nonmanifold_mesh(self, V, F):
        try:
            F_sorted = np.sort(F, axis=1)
            F_unique, indices = np.unique(F_sorted, axis=0, return_index=True)
            F = F[indices]

            mask = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
            F = F[mask]

            if igl.is_edge_manifold(F):
                return V, F
            else:
                return None, None
        
        except Exception as e:
            print(f"Error fixing non-manifold mesh: {e}")
            return None, None
    
    def _is_watertight(self, F):
        """Check if mesh is watertight (closed, no boundary)"""
        try:
            B = igl.boundary_loop(F)
            return len(B) == 0
        except:
            return False

    def _compute_bounds(self, V):
        """Compute bounding box"""
        return {
            'min': V.min(axis=0),
            'max': V.max(axis=0),
            'center': V.mean(axis=0),
            'size': V.max(axis=0) - V.min(axis=0)
        }
    
    def _compute_statistics(self):
        #Loading statistics
        if not self.meshes:
            return
        
        vertices = [m['vertices'] for m in self.meshes.values()]
        faces = [m['faces'] for m in self.meshes.values()]
        
        self.stats = {
            'total_meshes': len(self.meshes),
            'total_vertices': sum(vertices),
            'total_faces': sum(faces),
            'avg_vertices': np.mean(vertices),
            'avg_faces': np.mean(faces),
            'min_vertices': min(vertices),
            'max_vertices': max(vertices),
            'min_faces': min(faces),
            'max_faces': max(faces)
        }
    
    def print_statistics(self):
        print("\n=== Mesh Loading Statistics ===")
        print(f"Total meshes loaded: {self.stats.get('total_meshes', 0)}")
        print(f"Total vertices: {self.stats.get('total_vertices', 0):,}")
        print(f"Total faces: {self.stats.get('total_faces', 0):,}")
        print(f"Average vertices per mesh: {self.stats.get('avg_vertices', 0):.0f}")
        print(f"Average faces per mesh: {self.stats.get('avg_faces', 0):.0f}")
        print(f"Vertex range: {self.stats.get('min_vertices', 0):,} - {self.stats.get('max_vertices', 0):,}")
        print(f"Face range: {self.stats.get('min_faces', 0):,} - {self.stats.get('max_faces', 0):,}")
