import time
from collections import defaultdict
from scipy.spatial import Voronoi
from noise import snoise2
import numpy as np

start = time.time()
WIDTH = HEIGHT = 1000
points = np.random.randint(int(-WIDTH/10), int(WIDTH+WIDTH/10), size=(4000,2))

for i in range(2):
    vor = Voronoi(points)
    new_points = []
    for r in vor.regions:
        idx = [i for i in r if i >= 0]
        verts = vor.vertices[idx]
        if len(verts) > 0:
            new_points.append(verts.mean(axis=0))
    points = new_points

vor = Voronoi(points)

ridges = {i:set() for i in range(len(points))}
    

seed = np.random.randint(1000)

scale = 0.0007
elevation = (np.array([snoise2((x+seed)*scale, (y+seed)*scale, 4) for x,y in vor.vertices]) + 1)/2
water_line = np.percentile(elevation, 30)
river_score = np.zeros(len(vor.vertices))
basins = {}

def calc_slope():
    downhill = {i:(0,i) for i in range(len(vor.vertices))}
    for a,b in vor.ridge_vertices:
        if a != -1 and b != -1:
            low, high = sorted([a,b], key= lambda x: elevation[x])
            low_water = basins.get(verts_to_p[low], 0)
            high_water = basins.get(verts_to_p[high], 0)
            s = (elevation[high]+high_water)-(elevation[low]+low_water)
            if downhill[high][0] <= s:
                downhill[high] = (s, low)
    return downhill
verts_to_p = {v:pi for pi,p in enumerate(points)
                   for v in vor.regions[vor.point_region[pi]]
                   if v != -1}

peak_threshold = np.percentile(elevation, 70)
peak_verts = np.argwhere(elevation > peak_threshold).T[0]
for i in range(100):
    downhill = calc_slope()
    basin_verts = list({v for p in basins for v in vor.regions[p] if v != -1})*2
    choices = np.concatenate([peak_verts, list(basin_verts)])
    p = np.random.choice(choices)
    done = set()
    while True:
        done.add(p)
        river_score[p] += 1
        n = downhill[p][1]
        bp = verts_to_p[p]
        if n in done or bp in basins:
            basins[bp] = basins.get(bp, 0) + 0.5
            break
        p = n

river_score[np.argwhere(elevation < water_line).T[0]] = 0
river_score = np.sqrt(((river_score - river_score.min())/river_score.max())*20)

print("Done generating in {}".format(time.time()-start))

start = time.time()
import cairocffi as cairo

surface = cairo.ImageSurface (cairo.FORMAT_RGB24, WIDTH, HEIGHT)
ctx = cairo.Context(surface)

vertices = vor.vertices



for i, p in enumerate(points):
    vert_idx = vor.regions[vor.point_region[i]]
    if -1 not in vert_idx:
        verts = vertices[vert_idx]
        e = elevation[vert_idx].mean()
        if e < water_line:
            color = (0,0,0.6)
        elif i in basins:
            color = (0, 0, 1)
        elif e > peak_threshold:
            color = (e, e, e)
        else:
            color = (0,e,0)
        ctx.move_to (*verts[0])
        cx = verts[0][0]
        cy = verts[0][1]
        for x,y in verts[1:]:
            steps = 10
            dx = (x-cx)/steps
            dy = (y-cy)/steps
            for j in range(steps):
                ctx.line_to(cx+dx*steps+np.random.random(),cy+dy*steps+np.random.random())
            cx = x
            cy = y
        ctx.close_path()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1)
        ctx.stroke_preserve()
        ctx.fill()


for a,b in vor.ridge_vertices:
    if a != -1 and b != -1:
        river_width = river_score[[a,b]].min()
        (ax,ay), (bx,by) = vertices[[a,b]]
        ctx.move_to(ax, ay)
        ctx.line_to(bx, by)
        ctx.set_line_width(river_width)
        ctx.set_source_rgb(0, 0, 1)
        ctx.stroke()
surface.write_to_png('test.png')
print("Done drawing in {}".format(time.time()-start))
