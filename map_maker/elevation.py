import numpy as np

from noise import snoise2

def simplex_noise_elevation(mesh):
    seed = np.random.randint(10000)
    scale = 0.0007
    return (np.array([snoise2((x+seed)*scale, (y+seed)*scale, 4) for x,y in mesh.centers]) + 1)/2
