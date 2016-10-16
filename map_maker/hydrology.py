from heapq import heappush, heappop
import numpy as np

def planchon_darboux_fill(mesh):
    """Modify the mesh such that every non-edge node has a neighbor that is
    lower than itself.

    Using this simplifies water flow because water will never pool, however
    it makes lakes impossible and tends to form atrificial looking plains.
    """
    corrected_elevation = np.zeros(mesh.elevation.shape) + np.inf
    corrected_elevation[mesh.edge_regions] = np.minimum(0, mesh.elevation[mesh.edge_regions])
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

def calculate_flow(mesh, rainfall=80):
    """Calculates flux (the amount of water flowing through each node), slope
    (the difference in elevation between the node and it's lowest neighbor) and
    velocity (the speed at which it is flowing).

    Parameters
    ----------
    mesh : Mesh
        terrain to model on
    rainfall : float
        average rainfall in cm/year

    Returns
    -------
    flux : floats
        total water flowing through region in cm^3/year
    velocity : floats
         velocity of water flowing through region in meters/second
    """
    flux = np.zeros(len(mesh.elevation)+1)
    elevation = np.append(mesh.elevation + mesh.water, [np.inf])
    neighbors = np.hstack((mesh.neighbors, np.arange(len(mesh.neighbors))[np.newaxis].T))
    all_lowest = neighbors[np.arange(len(neighbors)), np.argmin(elevation[neighbors], axis=1)]

    downslope = mesh.centers[all_lowest]
    run = np.sqrt(np.power(downslope[:, 0] - mesh.centers[:, 0], 2) + np.power(downslope[:, 1] - mesh.centers[:, 1], 2))
    run = np.maximum(run, 0.0001)
    slope = (elevation[:-1]-elevation[all_lowest])/run
    # Based on TR-55's slope -> velocity chart for shallow concentrated flow
    # in meters per second
    velocity = 6.723453730104*np.power(slope,0.59099314)

    for i in np.argsort(elevation[:-1])[::-1]:
        e = elevation[i]
        lowest = all_lowest[i]
        flux[lowest] += flux[i] + rainfall * mesh.region_areas[i]
    flux = flux[:-1]
    return flux, velocity

def hydrolic_erosion(mesh):
    for _ in range(20):
       mesh.water = planchon_darboux_fill(mesh) - mesh.elevation
       mesh.water[mesh.water < 0.0001] = 0
       flux, velocity = calculate_flow(mesh)
       rate = velocity*np.sqrt(flux)
       mesh.elevation -= rate/200
    flux, velocity = calculate_flow(mesh)
    mesh.water_flux = flux
    return mesh

