import time
import math
from collections import defaultdict
from scipy.spatial import Voronoi, Delaunay
from noise import snoise2
import numpy as np

start = time.time()
WIDTH = HEIGHT = 1000
POLYGONS = 20000
points = np.random.randint(int(-WIDTH/10), int(WIDTH+WIDTH/10), size=(POLYGONS,2))

# Relax the mesh
for i in range(2):
    vor = Voronoi(points)
    new_points = []
    for r in vor.regions:
        idx = [i for i in r if i >= 0]
        verts = vor.vertices[idx]
        if len(verts) > 0:
            new_points.append(verts.mean(axis=0))
        points = new_points

mesh = Delaunay(Voronoi(points).vertices)

seed = np.random.randint(1000)

scale = 0.0007
centers = mesh.points[mesh.simplices].mean(axis=1)
elevation = (np.array([snoise2((x+seed)*scale, (y+seed)*scale, 4) for x,y in centers]) + 1)/2

# Do a  Planchon-Darboux fill
def planchon_darboux_fill(elevation, mesh):
    corrected_elevation = np.zeros(elevation.shape) + np.inf
    edges = mesh.neighbors.min(axis=1) == -1
    lowest = np.argmin(elevation)
    corrected_elevation[edges] = elevation[edges]
    did_change = True
    while did_change:
        did_change = False
        neighbors = mesh.neighbors[~edges]
        lowest = corrected_elevation[neighbors].min(axis=1)
        to_change = lowest < corrected_elevation[~edges]
        new_value = np.maximum(elevation[~edges][to_change], lowest[to_change]+0.0001)
        if np.sum(new_value != corrected_elevation[~edges][ to_change]) > 0:
            ne = corrected_elevation[~edges]
            ne[to_change] = new_value
            corrected_elevation[~edges]=ne
            did_change = True
    return corrected_elevation


def flux_and_slope(elevation, mesh):
    flux = np.zeros(elevation.shape)
    slope = np.zeros(elevation.shape)
    velocity = np.zeros(elevation.shape)
    for i, e in sorted(enumerate(elevation), key=lambda x:x[1], reverse=True):
        neighbors = [x for x in mesh.neighbors[i] if x >= 0]
        lowest = np.argmin(elevation[neighbors])
        flux[neighbors[lowest]] += flux[i]+1
        slope[i] = e - elevation[lowest]
        velocity[neighbors[lowest]] = (e - elevation[lowest]) + velocity[i]*0.9
    return flux/flux.max(), slope, velocity/velocity.max()


for _ in range(2):
    elevation = planchon_darboux_fill(elevation, mesh)
    flux, slope, velocity = flux_and_slope(elevation, mesh)
    elevation -= np.minimum(velocity*np.sqrt(flux), 0.1)

water_line = np.median(elevation)
flux = (flux+flux[elevation > water_line].min())/flux[elevation > water_line].max()
velocity = (velocity+velocity[elevation > water_line].min())/velocity[elevation > water_line].max()

# Smoothing
for _ in range(3):
    for i,e in enumerate(elevation):
        ne = elevation[[x for x in mesh.neighbors[i] if x != -1]]
        nwl = (ne > water_line).mean()
        if e > water_line and nwl < 0.5:
            elevation[i] = (e + ne.sum())/(len(ne)+1)
        elif e < water_line and nwl > 0.5:
            elevation[i] = (e + ne.sum())/(len(ne)+1)
elevation = planchon_darboux_fill(elevation, mesh)

print("Done generating in {}".format(time.time()-start))

start = time.time()
import cairocffi as cairo

surface = cairo.ImageSurface (cairo.FORMAT_RGB24, WIDTH, HEIGHT)
ctx = cairo.Context(surface)

ctx.set_line_width(1)
for i,(a,b,c) in enumerate(mesh.simplices):
    e = elevation[i]
    if e < water_line:
        color = (0, 0, 1)
    else:
        f = flux[i]
        v = velocity[i]*np.sqrt(f)
        color = (e,e,e)
    ctx.set_source_rgb(*color)
    ctx.move_to(mesh.points[a][0], mesh.points[a][1])
    ctx.line_to(mesh.points[b][0], mesh.points[b][1])
    ctx.line_to(mesh.points[c][0], mesh.points[c][1])
    ctx.close_path()
    ctx.stroke_preserve()
    ctx.fill()


ctx.set_source_rgb(0,0,0)
ctx.set_line_width(3)
for i,(a,b,c) in enumerate(mesh.simplices):
    e = elevation[i]
    if e >= water_line:
        nes = list(zip([(b,c),(c,a),(a,b)], elevation[mesh.neighbors[i]]))
        for (a,b), ve in nes:
            if ve < water_line:
                ctx.move_to(mesh.points[a][0], mesh.points[a][1])
                ctx.line_to(mesh.points[b][0], mesh.points[b][1])
                ctx.stroke()

downhill = np.zeros(elevation.shape, dtype=int)
for i, e in sorted(enumerate(elevation), key=lambda x:x[1], reverse=True):
    neighbors = [x for x in mesh.neighbors[i] if x >= 0]
    lowest = np.argmin(elevation[neighbors])
    downhill[i] = neighbors[lowest]

river_points = sorted(enumerate(flux), key=lambda x: x[1])
land_bits = set(np.array(range(len(elevation)))[elevation > water_line])
river_points = [i for i,_ in river_points if i in land_bits][-int(POLYGONS/10):]
ctx.set_line_width(2)
for i in river_points:
    verts = mesh.simplices[i]
    x,y = mesh.points[verts].mean(axis=0)
    line = [(x,y)]
    i = downhill[i]
    e = elevation[i]
    while e > water_line:
        verts = mesh.simplices[i]
        x,y = mesh.points[verts].mean(axis=0)
        line.append((x,y))
        i = downhill[i]
        e = elevation[i]

    previous = line[0]
    for i,current in enumerate(line[1:-1], 1):
        line[i] = np.mean(line[i-1:i+2], axis=0)
    ctx.move_to(*line[0])
    for x,y in line[1:]:
        ctx.line_to(x,y)
    ctx.stroke()
surface.write_to_png('test.png')
print("Done drawing in {}".format(time.time()-start))
