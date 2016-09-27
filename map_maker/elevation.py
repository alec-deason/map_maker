import math

import numpy as np
from scipy.misc import imsave

import cairocffi as cairo

from noise import snoise2

def simplex_noise_elevation(mesh):
    seed = np.random.randint(10000)
    scale = 0.0007
    return (np.array([snoise2((x+seed)*scale, (y+seed)*scale, 4) for x,y in mesh.centers]) + 1)/2

def cones_elevation(mesh):
    surface = cairo.ImageSurface (cairo.FORMAT_RGB24, mesh.width, mesh.height)
    ctx = cairo.Context(surface)

    pat = cairo.LinearGradient(np.random.random()*mesh.width, np.random.random()*mesh.height, np.random.random()*mesh.width, np.random.random()*mesh.height)
    pat.add_color_stop_rgba(0, 0.5, 0.5, 0.5, 1)
    pat.add_color_stop_rgba(1, 0, 0, 0, 1)
    ctx.set_source (pat)
    ctx.paint()

    for _ in range(30):
        x = np.random.random()*mesh.width
        y = np.random.random()*mesh.height
        r = np.random.random()*(np.min([x,y,mesh.width-x,mesh.height-y]))
        pat = cairo.RadialGradient (x, y, 0, x, y, r)
        pat.add_color_stop_rgba(0, 1, 1, 1, 1)
        pat.add_color_stop_rgba(1, 1, 1, 1, 0)

        ctx.set_source (pat)
        ctx.arc (x, y, r, 0, 2 * math.pi)
        ctx.fill()
    elevation = np.frombuffer(surface.get_data(), np.uint8)
    elevation = elevation.reshape((mesh.width, mesh.height, 4))[:,:, 0]
    cs = np.maximum(np.minimum(mesh.centers.astype(int), mesh.width-1), 0).astype(int)
    cx = cs[:, 0]
    cy = cs[:, 1]
    elevation = elevation[cx, cy]
    elevation = elevation / elevation.max() +  simplex_noise_elevation(mesh)
    return elevation
