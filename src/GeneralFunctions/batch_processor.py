import os
from shape_reader import read_mesh_file
from tqdm import tqdm
import os
import numpy as np

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

def load_meshes_with_progress(folder_path):
    import glob

    off_files = glob.glob(os.path.join(folder_path, "**/*.off"), recursive=True)
    ply_files = glob.glob(os.path.join(folder_path, "**/*.ply"), recursive=True)
    all_files = off_files + ply_files
    
    meshes = {}

    print(f"Found {len(all_files)} mesh files")
    
    for filepath in tqdm(all_files, desc="Loading meshes"):
        V, F, name = read_mesh_file(filepath)
        if V is not None and F is not None:
            meshes[name] = {'V': V, 'F': F}
    
    return meshes

def load_meshes_cached(folder_path, cache_file="meshes_cache.npz", force_reload=False):
    npz_file = cache_file.replace('.pkl', '.npz')
    
    if not force_reload and os.path.exists(npz_file):
        cache_time = os.path.getmtime(npz_file)
        folder_time = os.path.getmtime(folder_path)
        
        if cache_time > folder_time:
            print("Loading meshes from cache...")
            import time
            t0 = time.time()
            data = np.load(npz_file, allow_pickle=False)
            meshes = {}
            # Keys are stored as "name__V" and "name__F"
            names = set(k.rsplit('__', 1)[0] for k in data.files)
            for name in names:
                meshes[name] = {
                    'V': data[f'{name}__V'],
                    'F': data[f'{name}__F'],
                }
            print(f"Loaded {len(meshes)} meshes in {time.time() - t0:.2f}s")
            return meshes

    print("Loading meshes from disk...")
    meshes = load_all_meshes(folder_path)
    
    # Save as npz
    arrays = {}
    for name, mesh in meshes.items():
        safe = name.replace('/', '_').replace('\\', '_')
        arrays[f'{safe}__V'] = mesh['V']
        arrays[f'{safe}__F'] = mesh['F']
    np.savez(npz_file, **arrays)
    print(f"Saved mesh cache to {npz_file}")
    return meshes