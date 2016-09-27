import math

import numpy as np

from scipy.ndimage.filters import gaussian_filter

class Cities:
    def __init__(self):
        self.seeds = []

def city_score(mesh, locations, target_count):
    spacing = np.mean([mesh.width,mesh.height])/target_count
    mask = np.logical_or(np.logical_or(mesh.centers[:,0] < spacing, np.logical_or(np.logical_or(mesh.elevation < mesh.water_line, mesh.centers[:,0] >= mesh.width-spacing), mesh.centers[:,1] >= mesh.height-spacing)), mesh.centers[:,1] < spacing)
    water_proximity = np.zeros(mesh.elevation.shape)
    water_proximity[mesh.elevation < mesh.water_line] = 1
    for _ in range(10):
        water_proximity += water_proximity[mesh.neighbors].mean(axis=1)
    
    score = np.zeros(mesh.elevation.shape, dtype=float)
    score[mask] = -np.inf
    score += water_proximity
    score += mesh.water_flux*4
    for i in locations:
        x,y = mesh.centers[i]
        dists = np.sqrt(np.power(mesh.centers[:,0]-x, 2)+np.power(mesh.centers[:,1]-y, 2))
        score[dists < spacing] = -np.inf
    return score

def place_seeds(mesh, count):
    locations = []
    mesh.population = np.zeros(mesh.elevation.shape)
    for _ in range(count):
        score = city_score(mesh, locations, count)
        i = np.argmax(score)
        locations.append(i)
        mesh.population[i] = 150
    return mesh

def grow_population(mesh, target_total):
    scale = ((mesh.width*mesh.height)/len(mesh.elevation))*0.05
    while mesh.population.sum() < target_total:
        l = np.random.choice(len(mesh.centers), p=mesh.population/mesh.population.sum())
        d = np.abs(np.random.standard_cauchy()) * scale
        a = np.random.random()*math.pi*2
        x = np.cos(a)*d+mesh.centers[l][0]
        y = np.sin(a)*d+mesh.centers[l][1]
        i = mesh.point_to_region((x,y))
        if mesh.elevation[i] > mesh.water_line:
            mesh.population[i] += 150
            if mesh.population[i] > 1000:
                ns = [n for n in mesh.neighbors[i] if n >= 0 and mesh.elevation[n] > mesh.water_line]
                mesh.population[i] = 1000
                if ns:
                    overflow = mesh.population[i] - 1000
                    mesh.population[ns] += overflow/len(ns)


    print(mesh.population.min(), np.median(mesh.population), mesh.population.max())
    return mesh

