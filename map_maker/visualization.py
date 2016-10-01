import math
import numpy as np

import cairocffi as cairo

def draw(path, mesh, width, height):
    surface = cairo.ImageSurface (cairo.FORMAT_RGB24, width, height)
    ctx = cairo.Context(surface)
    ctx.scale(width/mesh.width, height/mesh.height)
    ctx.set_line_width(1)
    water_color = np.power(1-mesh.water/mesh.water.max(), 2)
    for i,(verts) in enumerate(mesh.regions):
        e = mesh.elevation[i]
        e += mesh.water[i]
        color = (e,e,e)
        if mesh.water[i] > 0:
            color = (0, 0, water_color[i])
        else:
            color = (e*.9,e*.8,e*.7)
        ctx.move_to(*mesh.points[verts[0]])
        for v in verts[1:]:
            ctx.line_to(mesh.points[v][0], mesh.points[v][1])
        ctx.close_path()
        ctx.set_source_rgb(*color)
        ctx.stroke_preserve()
        ctx.fill()


    ctx.set_source_rgb(0,0,0)
    ctx.set_line_width(1)
    for i,(a,b,c) in enumerate(mesh.regions):
        w = mesh.water[i]
        if w == 0:
            nes = list(zip([(b,c),(c,a),(a,b)], mesh.water[mesh.neighbors[i]]))
            for (a,b), ve in nes:
                if ve > 0:
                    ctx.move_to(mesh.points[a][0], mesh.points[a][1])
                    ctx.line_to(mesh.points[b][0], mesh.points[b][1])
                    ctx.stroke()

    ctx.set_source_rgb(1,0,0)
    ctx.set_line_width(1)
    for i,(a,b,c) in enumerate(mesh.regions):
        p = mesh.population[i]
        if p > 0:
            nps = list(zip([(b,c),(c,a),(a,b)], mesh.population[mesh.neighbors[i]]))
            for (a,b), vp in nps:
                if vp <= 0:
                    ctx.move_to(mesh.points[a][0], mesh.points[a][1])
                    ctx.line_to(mesh.points[b][0], mesh.points[b][1])
                    ctx.stroke()
    downhill = np.zeros(mesh.elevation.shape, dtype=int)
    for i, e in sorted(enumerate(mesh.elevation), key=lambda x:x[1], reverse=True):
        neighbors = [x for x in mesh.neighbors[i] if x >= 0]
        lowest = np.argmin(mesh.elevation[neighbors])
        downhill[i] = neighbors[lowest]

    ctx.set_source_rgb(0,0,0)
    ctx.set_line_width(1)
    river_points = sorted(enumerate(mesh.water_flux), key=lambda x: x[1])
    land_bits = set(np.array(range(len(mesh.elevation)))[mesh.water == 0])
    river_points = [i for i,_ in river_points if i in land_bits][-int(len(mesh.elevation)/10):]
    for i in river_points:
        x,y = mesh.centers[i]
        line = [(x,y)]
        done = {i}
        i = downhill[i]
        w = mesh.water[i]
        while w == 0 and i not in done:
            done.add(i)
            x,y = mesh.centers[i]
            line.append((x,y))
            i = downhill[i]
            w = mesh.water[i]

        previous = line[0]
        fluxes = []
        ctx.set_line_width(1)
        ctx.move_to(*line[0])
        for x,y in line[1:]:
            ctx.line_to(x,y)
        ctx.stroke()

    surface.write_to_png(path)
