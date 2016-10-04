from heapq import heappush, heappop
import numpy as np

def planchon_darboux_fill(mesh):
    """Modify the mesh such that every non-edge node has a neighbor that is
    lower than itself.

    Using this simplifies water flow because water will never pool, however
    it makes lakes impossible and tends to form atrificial looking plains.
    """
    corrected_elevation = np.zeros(mesh.elevation.shape) + np.inf
    level = np.percentile(mesh.elevation, 30)
    low_edges = np.logical_and(mesh.edge_regions, mesh.elevation < level)
    corrected_elevation[mesh.edge_regions] = mesh.elevation[mesh.edge_regions]
    corrected_elevation[low_edges] = level
    did_change = True
    while did_change:
        did_change = False
        neighbors = mesh.neighbors[~mesh.edge_regions]
        lowest = corrected_elevation[neighbors].min(axis=1)
        to_change = lowest < corrected_elevation[~mesh.edge_regions]
        new_value = np.maximum(mesh.elevation[~mesh.edge_regions][to_change], lowest[to_change]+0.00001)
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
    velocity = np.zeros(len(mesh.elevation)+1)
    elevation = np.append(mesh.elevation, [np.inf])
    #elevation = np.append(mesh.elevation + mesh.water, [np.inf])
    all_lowest = mesh.neighbors[np.arange(len(mesh.neighbors)), np.argmin(elevation[mesh.neighbors], axis=1)]

    for i in np.argsort(elevation[:-1])[::-1]:
        e = elevation[i]
        lowest = all_lowest[i]
        flux[lowest] += flux[i]+1
        velocity[mesh.neighbors[lowest]] = (e - elevation[lowest]) + velocity[i]*0.9
    flux = flux[:-1]
    velocity = velocity[:-1]
    return (flux/flux[mesh.water == 0].max()), (velocity/velocity[mesh.water ==0].max())

def hydrolic_erosion(mesh):
    for _ in range(20):
        mesh.water = planchon_darboux_fill(mesh) - mesh.elevation
        mesh.water[mesh.water < 0.0001] = 0
        flux, velocity = calculate_flow(mesh)
        rate = velocity*np.sqrt(flux)
        rate = rate.clip(0, np.percentile(rate, 90))
        rate /= rate.max()
        rate *= 0.0025
        mesh.elevation -= rate
    flux, velocity = calculate_flow(mesh)
    mesh.water_flux = flux
    print(flux.min(), flux.max(), flux.mean())
    mesh.elevation = (mesh.elevation - mesh.elevation.min()) / mesh.elevation.max()
    return mesh

