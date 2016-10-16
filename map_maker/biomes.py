import numpy as np

def temperature(mesh):
    """Mean annual temperature. Currently assumes a temperate latitude
    
    Returns
    -------
    float
    Mean annual temperature in C
    """
    norm_el = mesh.elevation - np.percentile(mesh.elevation[mesh.water == 0], 2)
    norm_el /= norm_el.max()
    minimum = 0
    maximum = 24
    return 1-norm_el

def precipitation(mesh):
    """Silly model that just accounts for proximity to water

    Returns
    -------
    float
    mm of precipitation per year
    """
    surface_water = mesh.water_flux*50 + (mesh.water > 0).astype(int)*10
    hum = np.zeros(surface_water.shape)
    for _ in range(40):
        hum += surface_water
        hum = hum[mesh.neighbors].mean(axis=1)
    minimum = 62.5
    maximum = 16000
    hum = hum/hum.max()
    hum *= maximum-62.5
    hum += 62.5
    return hum

def life_zones(mesh):
    """Assign biomes based on https://en.wikipedia.org/wiki/Holdridge_life_zones
    """
    pass
    

def assign_biomes(mesh):
    biome_names = ['temperate_forest', 'tropical_forest', 'desert', 'swamp', 'beach', 'water', 'ice']
    biome_map = {n:i for i,n in enumerate(biome_names)}

    ice = mesh.temperature < np.percentile(mesh.temperature, 5)
    #beach = np.any(mesh.water[mesh.neighbors] > 0, axis=1)
    #beach = np.logical_and(beach, ~ice)
    beach = np.zeros(ice.shape) > 0
    nib = np.logical_and(~beach, ~ice)

    desert = mesh.precipitation < np.percentile(mesh.precipitation[mesh.water == 0], 5)
    desert = np.logical_and(nib, desert)

    swamp = mesh.precipitation > np.percentile(mesh.precipitation[mesh.water == 0], 95)
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
