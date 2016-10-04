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
    aspect = mesh.width / mesh.height
    width = 500
    height = int(width / aspect)
    scale = width/mesh.width
    surface = cairo.ImageSurface (cairo.FORMAT_RGB24, width, height)
    ctx = cairo.Context(surface)

    pat = cairo.LinearGradient(width/2, 0, width/2, height)
    pat.add_color_stop_rgba(0, 1, 1, 1, 1)
    pat.add_color_stop_rgba(1, 0, 0, 0, 1)
    ctx.set_source (pat)
    ctx.paint()

    for _ in range(30):
        x = np.random.random()*width
        y = np.random.random()*height
        r = np.random.random()*(np.min([x,y,width-x,height-y]))
        pat = cairo.RadialGradient (x, y, 0, x, y, r)
        pat.add_color_stop_rgba(0, 1, 1, 1, 1)
        pat.add_color_stop_rgba(1, 1, 1, 1, 0)

        ctx.set_source (pat)
        ctx.arc (x, y, r, 0, 2 * math.pi)
        ctx.fill()
    elevation = np.frombuffer(surface.get_data(), np.uint8)
    elevation = elevation.reshape((width, height, 4))[:,:, 0]
    cx = np.maximum(np.minimum(mesh.centers[:, 0], mesh.width-1), 0)
    cy = np.maximum(np.minimum(mesh.centers[:, 1], mesh.height-1), 0)
    cx = (cx*scale).astype(int)
    cy = (cy*scale).astype(int)
    elevation = elevation[cx, cy]
    elevation = elevation / elevation.max() *  simplex_noise_elevation(mesh)
    elevation[mesh.edge_regions] = elevation[~mesh.edge_regions].mean()
    return elevation
