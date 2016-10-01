import numpy as np

from scipy.spatial import Voronoi, Delaunay

class Mesh:
    def __init__(self):
        self.width = None
        self.height = None
        self.centers = None
        self.vertices = None
        self.regions = None
        self.edge_regions = None
        self.neighbors = None
        self.elevation = None
        self.water = None
        self.water_flux = None
        self.population = None
        self.point_to_region = None

def delauny_mesh(width, height, point_count):
    points = np.random.randint(int(-width/10), int(width+height/10), size=(point_count,2))

    # Relax the mesh
    for i in range(2):
        vor = Voronoi(points)
        new_points = []
        for r in vor.regions:
            idx = [i for i in r if i >= 0]
            verts = vor.vertices[idx]
            if len(verts) > 0:
                new_points.append(verts.mean(axis=0))
            points = new_points

    dmesh = Delaunay(Voronoi(points).vertices)
    mesh = Mesh()
    mesh.width = width
    mesh.height = height
    mesh.centers = dmesh.points[dmesh.vertices].mean(axis=1)
    mesh.points = dmesh.points
    mesh.regions = dmesh.simplices
    mesh.vertices = dmesh.vertices
    mesh.edge_regions = np.logical_or(mesh.centers[:,0] > width, 
                        np.logical_or(mesh.centers[:,1] > height,
                        dmesh.neighbors.min(axis=1) == -1))
    mesh.neighbors = dmesh.neighbors
    mesh.point_to_region = dmesh.find_simplex

    return mesh
