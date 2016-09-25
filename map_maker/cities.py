import numpy as np

class Cities:
    def __init__(self):
        self.seeds = []

def city_score(mesh, locations, target_count):
    spacing = np.mean([mesh.width,mesh.height])/target_count
    mask = np.logical_or(np.logical_or(mesh.centers[:,0] < spacing, np.logical_or(np.logical_or(mesh.elevation < mesh.water_line, mesh.centers[:,0] >= mesh.width-spacing), mesh.centers[:,1] >= mesh.height-spacing)), mesh.centers[:,1] < spacing)
    score = np.zeros(mesh.elevation.shape, dtype=float)
    score[mask] = -np.inf
    score += mesh.water_flux*4
    for i in locations:
        x,y = mesh.centers[i]
        dists = np.sqrt(np.power(mesh.centers[:,0]-x, 2)+np.power(mesh.centers[:,1]-y, 2))
        score[dists < spacing] = -np.inf
    return score

def place_seeds(mesh, count):
    locations = []
    while len(locations) < count:
        score = city_score(mesh, locations, count)
        locations.append(np.argmax(score))
    cities = Cities()
    cities.seeds = locations
    return cities

