import numpy as np

def planchon_darboux_fill(mesh):
    """Modify the mesh such that every non-edge node has a neighbor that is
    lower than itself.

    Using this simplifies water flow because water will never pool, however
    it makes lakes impossible and tends to form atrificial looking plains.
    """
    corrected_elevation = np.zeros(mesh.elevation.shape) + np.inf
    edge_nodes = mesh.neighbors.min(axis=1) == -1
    corrected_elevation[edge_nodes] = mesh.elevation[edge_nodes]
    did_change = True
    while did_change:
        did_change = False
        neighbors = mesh.neighbors[~edge_nodes]
        lowest = corrected_elevation[neighbors].min(axis=1)
        to_change = lowest < corrected_elevation[~edge_nodes]
        new_value = np.maximum(mesh.elevation[~edge_nodes][to_change], lowest[to_change]+0.0001)
        if np.sum(new_value != corrected_elevation[~edge_nodes][to_change]) > 0:
            ne = corrected_elevation[~edge_nodes]
            ne[to_change] = new_value
            corrected_elevation[~edge_nodes] = ne
            did_change = True
    return corrected_elevation

def calculate_flow(mesh):
    """Calculates flux (the amount of water flowing through each node), slope
    (the difference in elevation between the node and it's lowest neighbor) and
    velocity (the speed at which it is flowing).
    """
    flux = np.zeros(mesh.elevation.shape)
    slope = np.zeros(mesh.elevation.shape)
    velocity = np.zeros(mesh.elevation.shape)
    for i, e in sorted(enumerate(mesh.elevation), key=lambda x:x[1], reverse=True):
        neighbors = [x for x in mesh.neighbors[i] if x >= 0]
        lowest = np.argmin(mesh.elevation[neighbors])
        flux[neighbors[lowest]] += flux[i]+1
        slope[i] = e - mesh.elevation[lowest]
        velocity[neighbors[lowest]] = (e - mesh.elevation[lowest]) + velocity[i]*0.9
    return flux/flux.max(), slope, velocity/velocity.max()

def smooth_coast_lines(mesh):
    """Reduce erosion artifacts by lowering small, sharp islands and filling in
    isolated bits.
    """
    water_line = np.median(mesh.elevation)
    elevation = mesh.elevation.copy()
    for _ in range(3):
        for i,e in enumerate(elevation):
            ne = elevation[[x for x in mesh.neighbors[i] if x != -1]]
            nwl = (ne > water_line).mean()
            if e > water_line and nwl < 0.5:
                elevation[i] = (e + ne.sum())/(len(ne)+1)
            elif e < water_line and nwl > 0.5:
                elevation[i] = (e + ne.sum())/(len(ne)+1)
    return elevation

def hydrolic_erosion(mesh):
    for _ in range(2):
        mesh.elevation = planchon_darboux_fill(mesh)
        flux, sloke, velocity = calculate_flow(mesh)
        mesh.elevation -= np.minimum(velocity*np.sqrt(flux), 0.01)
    mesh.elevation = smooth_coast_lines(mesh)
    flux, sloke, velocity = calculate_flow(mesh)
    mesh.water_flux = flux
    mesh.water_line = np.median(mesh.elevation)
    return mesh

