import os
from src.GeneralFunctions.shape_reader import read_mesh_file


def load_all_meshes(folder_path, supported_extensions=['.off', '.ply', '.obj', '.stl', '.mesh']):
    meshes = {}
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    mesh_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                mesh_files.append(os.path.join(root, file))
    
    print(f"Found {len(mesh_files)} mesh files")

    for i, filepath in enumerate(mesh_files):
        print(f"Reading {i+1}/{len(mesh_files)}: {os.path.basename(filepath)}")
        
        V, F, mesh_name = read_mesh_file(filepath)
        
        if V is not None and F is not None:
            # Handle duplicate names
            if mesh_name in meshes:
                # Append number to make unique
                counter = 1
                while f"{mesh_name}_{counter}" in meshes:
                    counter += 1
                mesh_name = f"{mesh_name}_{counter}"
            
            meshes[mesh_name] = {
                'V': V,
                'F': F,
                'filepath': filepath,
                'num_vertices': V.shape[0],
                'num_faces': F.shape[0]
            }
        else:
            print(f"Failed to read: {filepath}")

    print(f"Successfully loaded {len(meshes)} meshes")
    return meshes
        