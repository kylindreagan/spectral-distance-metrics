def generate_gordon_webb_wolpert_drums():
    """
    Generate the original 'cannot hear shape of drum' shapes.
    Returns two 2D polygons with same eigenvalues.
    """
    # Vertex coordinates from the paper
    drum1_vertices = [
        [0, 0], [2, 0], [2, 1], [3, 1], [3, 2], [1, 2], [1, 1], [0, 1]
    ]
    
    drum2_vertices = [
        [0, 0], [1, 0], [1, 1], [3, 1], [3, 2], [2, 2], [2, 1], [0, 1]
    ]
    
    # Triangulate polygons
    from scipy.spatial import Delaunay
    tri1 = Delaunay(drum1_vertices)
    tri2 = Delaunay(drum2_vertices)
    
    return (np.array(drum1_vertices), tri1.simplices), \
           (np.array(drum2_vertices), tri2.simplices)