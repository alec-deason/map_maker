import os.path
from time import time

import numpy as np

import scipy
from scipy.misc import imsave

from noise import snoise2, snoise3

MAX_ELEVATION = 100000
WIDTH = HEIGHT = 300
DEPTH = 30
MATERIALS = np.array([10000]+[x*20 for x in range(200, 1,-5)])


def perlin_base_terrain():
    terrain = np.ndarray((WIDTH, HEIGHT))
    material = np.zeros((WIDTH, HEIGHT, DEPTH))
    material += 1000
    octaves = 16
    freq = 32.0 * octaves

    mat_octaves = 8
    mat_freq = 64.0 * mat_octaves
    for y in range(HEIGHT):
        for x in range(WIDTH):
            terrain[y,x] = snoise2(x / freq, y / freq, octaves)
            for z in range(DEPTH):
                material[y, x, z] = snoise3(x / mat_freq, y / mat_freq, z / mat_freq, mat_octaves)
    terrain /= 2
    terrain *= MAX_ELEVATION
    terrain += MAX_ELEVATION/2

    material /= 2
    material += 0.5
    material *= len(MATERIALS) - 1
    material += 1
    material = MATERIALS[material.astype(int)]

    imsave('data/elevation.png', terrain)
    return terrain, material

def neighboor_slopes(elevation, water, water_line):
    te = elevation+water
    te = np.pad(te, 1, 'constant', constant_values=[water_line])
    width, height = elevation.shape
    points_y, points_x = np.mgrid[0:width, 0:height]
    slopes = np.ndarray((9, width, height))
    next_points_x = np.stack([points_x]*9)
    next_points_y = np.stack([points_y]*9)
    i = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            np_x = points_x + x + 1
            np_y = points_y + y + 1
            slopes[i] = np.maximum(te[1:-1,1:-1] - te[np_y, np_x], 0)
            next_points_x[i] = np_x
            next_points_y[i] = np_y
            i += 1
    sum_slopes = slopes.sum(axis=0)
    outflow = np.minimum(sum_slopes, water)
    slopes[:, sum_slopes != 0] /= sum_slopes[sum_slopes != 0]
    slopes[:, sum_slopes == 0] = 0
    return slopes, outflow, next_points_x, next_points_y

def gradient_water(elevation, material):
    width, height = elevation.shape
    water = np.zeros(elevation.shape, dtype=np.float64)
    carrying = np.zeros(elevation.shape, dtype=np.float64)
    sediment = np.zeros(elevation.shape, dtype=np.float64)
    rain_rate = 150
    points_y, points_x = np.mgrid[0:width, 0:height]
    water_line = np.percentile(elevation, 20)
    water[elevation <= water_line] = water_line - elevation[elevation <= water_line]
    water_thresh = rain_rate*2
    initial_water = water.sum()
    rain =0
    durations = []
    initial_material=elevation.sum()
    for iteration in range(130):
        start = time()
        if iteration%10 ==0:
            print(iteration, np.mean(durations))
            print(carrying.sum()/initial_material, carrying.sum()/water.sum())
        rain = (initial_water - water.sum()) / (width*height)
        water = np.maximum(0, water-rain_rate/2)
        water += np.random.random(size=water.shape)*rain

        slopes, outflow, next_points_x, next_points_y = neighboor_slopes(elevation+sediment, water, MAX_ELEVATION+1000)

        new_water = np.pad(water, 1, 'constant', constant_values=[0])

        np.add.at(new_water, [next_points_y, next_points_x], slopes*outflow*0.1)
        new_water = new_water[1:-1, 1:-1]
        flow = np.nan_to_num(outflow/water)


        #drop = np.minimum((carrying * (1-flow)), water*.5)
        drop = carrying * (1-flow) * 0.5
        elevation += drop

        carrying -= drop
        new_water = np.maximum(0, new_water-drop)
        agro = material[points_y, points_x, (elevation[points_y, points_x]/MAX_ELEVATION).astype(int) * DEPTH]

        pickup = np.minimum(elevation, agro*(outflow/(rain_rate*100))*flow)
        elevation -= pickup
        carrying += pickup

        new_water += pickup

        new_water -= outflow*0.1

        water = new_water

        durations.append(time()-start)
    np.save('elevation.npy', elevation)
    np.save('water.npy', water)

if __name__ == '__main__':
    data_path = 'data'
    elevation, material = perlin_base_terrain()

    print('Running water')
    gradient_water(elevation, material)

