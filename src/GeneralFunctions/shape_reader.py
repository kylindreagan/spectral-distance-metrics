import os
import numpy as np
import igl

def read_mesh_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.off':
            V, F = igl.readOFF(filepath)
        elif ext == '.ply':
            V, F = igl.read_triangle_mesh(filepath)
        elif ext == '.obj':
            V, F = igl.readOBJ(filepath)
        elif ext == '.stl':
            V, F = igl.read_triangle_mesh(filepath)
        elif ext == '.mesh':
            V, F = igl.readMESH(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Ensure V is n x 3 and F is m x 3
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)

        # Remove degenerate faces if any
        if F.size > 0:
            F = F[F[:, 0] != F[:, 1]]
            F = F[F[:, 1] != F[:, 2]]
            F = F[F[:, 2] != F[:, 0]]

        mesh_name = os.path.splitext(os.path.basename(filepath))[0]
        return V, F, mesh_name
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None