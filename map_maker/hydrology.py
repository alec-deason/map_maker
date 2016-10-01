from heapq import heappush, heappop
import numpy as np

def planchon_darboux_fill(mesh):
    """Modify the mesh such that every non-edge node has a neighbor that is
    lower than itself.

    Using this simplifies water flow because water will never pool, however
    it makes lakes impossible and tends to form atrificial looking plains.
    """
    corrected_elevation = np.zeros(mesh.elevation.shape) + np.inf
    corrected_elevation[mesh.edge_regions] = np.median(mesh.elevation)#mesh.elevation[mesh.edge_regions]
    did_change = True
    while did_change:
        did_change = False
        neighbors = mesh.neighbors[~mesh.edge_regions]
        lowest = corrected_elevation[neighbors].min(axis=1)
        to_change = lowest < corrected_elevation[~mesh.edge_regions]
        new_value = np.maximum(mesh.elevation[~mesh.edge_regions][to_change], lowest[to_change]+0.0001)
        if np.sum(new_value != corrected_elevation[~mesh.edge_regions][to_change]) > 0:
            ne = corrected_elevation[~mesh.edge_regions]
            ne[to_change] = new_value
            corrected_elevation[~mesh.edge_regions] = ne
            did_change = True
    return corrected_elevation

def calculate_flow(mesh):
    """Calculates flux (the amount of water flowing through each node), slope
    (the difference in elevation between the node and it's lowest neighbor) and
    velocity (the speed at which it is flowing).
    """
    flux = np.zeros(len(mesh.elevation)+1)
    slope = np.zeros(len(mesh.elevation)+1)
    velocity = np.zeros(len(mesh.elevation)+1)
    elevation = np.append(mesh.elevation, [np.inf])
    #water = np.zeros(len(mesh.elevation)+1)
    all_lowest = mesh.neighbors[np.arange(len(mesh.neighbors)), np.argmin(elevation[mesh.neighbors], axis=1)]
    #filling = True
    #water_covered_count = 0
    #while filling:
    #    diff = np.append(elevation[all_lowest] - elevation[:-1], [-1])
    #    basins = diff >= 0
    #    water[basins] += diff[basins] + 0.0001
    #    elevation[basins] += diff[basins] + 0.0001
    #    filling = np.percentile(elevation, 30) < np.median(mesh.elevation)
    #    print(water.max())
    #    all_lowest = mesh.neighbors[np.arange(len(mesh.neighbors)), np.argmin(elevation[mesh.neighbors], axis=1)]

    slope = mesh.elevation - mesh.elevation[all_lowest]
    for i in np.argsort(elevation[:-1])[::-1]:
        e = elevation[i]
        lowest = all_lowest[i]
        flux[lowest] += flux[i]+1
        velocity[mesh.neighbors[lowest]] = (e - mesh.elevation[lowest]) + velocity[i]*0.9
    return (flux/flux.max())[:-1], slope[:-1], (velocity/velocity.max())[:-1]#, water[:-1]

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
    for _ in range(5):
        mesh.water = planchon_darboux_fill(mesh) - mesh.elevation
        flux, sloke, velocity = calculate_flow(mesh)
        mesh.elevation -= np.minimum(velocity*np.sqrt(flux), 0.05)
    #mesh.elevation = smooth_coast_lines(mesh)
    flux, sloke, velocity = calculate_flow(mesh)
    mesh.water_flux = flux
    mesh.elevation = (mesh.elevation - mesh.elevation.min()) / mesh.elevation.max()
    mesh.water_line = np.median(mesh.elevation)
    return mesh

