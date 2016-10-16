import math

import numpy as np
import scipy.stats
from scipy.misc import imsave

import cairocffi as cairo

from noise import snoise2

MAX_ELEVATION = 9500

def simplex_noise_elevation(mesh, mean=0.5, std=0.1):
    seed = np.random.randint(10000)
    scale = 0.0007
    elevation = np.array([snoise2((x+seed)*scale, (y+seed)*scale, 4) for x,y in mesh.centers])
    precentiles = scipy.stats.mstats.rankdata(elevation)/len(elevation)
    precentiles[precentiles == 1] = 0.99
    return scipy.stats.norm.ppf(precentiles, mean, std)

def cones_elevation(mesh, plateau=0, cone_count=3, mountain_height_mean=3771, mountain_height_std=1907):
    """Create an elevation map by drawing a base plane and then randomly distributing cones over it.

    Parameters
    ----------
    mesh : Mesh
        Mesh to apply the elevation to
    plateau : float
        Elevation of the base plane in meters
    cone_count : int
        Number of cones to place
    mountain_height_mean : float
        mean of the distribution to draw mountain heights from in meters from sea level
    mountain_height_std : float
        standard deviation of the distribution to draw mountain heights from in meters from sea level
    """


    aspect = mesh.width / mesh.height
    width = 500
    height = int(width / aspect)
    scale = width/mesh.width

    cone_heights = np.random.normal(mountain_height_mean, mountain_height_std, cone_count)
    cone_centers_x = np.random.random(cone_count)*width*2 - width/2
    cone_centers_y = np.random.random(cone_count)*height*2 - height/2

    surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    ctx.set_source_rgb(0,0,0)
    ctx.paint()

    for x, y, h in zip(cone_centers_x, cone_centers_y, cone_heights):
        r = h * scale
        pat = cairo.RadialGradient (x, y, 0, x, y, r)
        pat.add_color_stop_rgba(0, 1, 1, 1, h/MAX_ELEVATION)
        pat.add_color_stop_rgba(1, 1, 1, 1, 0)

        ctx.set_source (pat)
        ctx.arc (x, y, r, 0, 2 * math.pi)
        ctx.fill()
    elevation = np.frombuffer(surface.get_data(), np.uint8)
    elevation = elevation.reshape((width, height, 4))[:,:, 0].astype(float)
    elevation /= 255
    cx = np.maximum(np.minimum(mesh.centers[:, 0], mesh.width-1), 0)
    cy = np.maximum(np.minimum(mesh.centers[:, 1], mesh.height-1), 0)
    cx = (cx*scale).astype(int)
    cy = (cy*scale).astype(int)
    elevation = elevation[cx, cy] * MAX_ELEVATION
    elevation += simplex_noise_elevation(mesh, plateau, mountain_height_std/100)
    elevation[mesh.edge_regions] = elevation[~mesh.edge_regions].mean()
    return elevation
