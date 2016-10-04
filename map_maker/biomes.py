import numpy as np

def temperature(mesh):
    norm_el = mesh.elevation - np.percentile(mesh.elevation[mesh.water == 0], 2)
    norm_el /= norm_el.max()
    return 1-norm_el

def humidity(mesh):
    surface_water = mesh.water_flux*50 + (mesh.water > 0).astype(int)*10
    hum = np.zeros(surface_water.shape)
    for _ in range(40):
        hum += surface_water
        hum = hum[mesh.neighbors].mean(axis=1)
    return hum/hum.max()

def assign_biomes(mesh):
    biome_names = ['ice', 'temperate_forest', 'tropical_forest', 'desert', 'swamp', 'beach', 'water']
    biome_map = {n:i for i,n in enumerate(biome_names)}

    ice = mesh.temperature < np.percentile(mesh.temperature, 5)
    #beach = np.any(mesh.water[mesh.neighbors] > 0, axis=1)
    #beach = np.logical_and(beach, ~ice)
    beach = np.zeros(ice.shape) > 0
    nib = np.logical_and(~beach, ~ice)

    desert = mesh.humidity < np.percentile(mesh.humidity[mesh.water == 0], 5)
    desert = np.logical_and(nib, desert)

    swamp = mesh.humidity > np.percentile(mesh.humidity[mesh.water == 0], 95)
    swamp = np.logical_and(nib, swamp)

    biomes = np.zeros(mesh.elevation.shape) - 1
    
    biomes[ice] = biome_map['ice']
    biomes[desert] = biome_map['desert']
    biomes[beach] = biome_map['beach']
    biomes[swamp] = biome_map['swamp']

    biomes[np.logical_and(biomes < 0, mesh.temperature > np.percentile(mesh.temperature[mesh.water == 0], 50))] = biome_map['tropical_forest']
    biomes[biomes < 0] = biome_map['temperate_forest']
    biomes[np.logical_and(~ice, mesh.water > 0)] = biome_map['water']

    biome_ids = {v:k for k,v in biome_map.items()}
    return biomes, biome_ids
