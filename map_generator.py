from map_maker.mesh import delauny_mesh
from map_maker.elevation import simplex_noise_elevation, cones_elevation
from map_maker.hydrology import hydrolic_erosion
from map_maker.cities import place_seeds, grow_population
from map_maker.visualization import draw, draw_matplotlib
from map_maker import biomes

import pickle
import sys
import numpy as np

def main():
    mesh = delauny_mesh(10000, 10000, 10000)
    print('Mesh generated')
    mesh.elevation = cones_elevation(mesh, plateau=1000, cone_count=5)
    print('Initial elevation calculated')
    #TODO: What's the right interface here? These should be consistent.
    # Either modify the mesh in place or return a modified copy.
    hydrolic_erosion(mesh)
    mesh.temperature = biomes.temperature(mesh)
    mesh.precipitation = biomes.precipitation(mesh)
    mesh.biomes, mesh.biome_ids = biomes.assign_biomes(mesh)
    print('Erosion finished')
    mesh = place_seeds(mesh, 15)
    print('Cities seeded')
    #mesh = grow_population(mesh, 100000)
    #print('Population distributed')
    with open('mesh.pickle', 'wb') as f:
        pickle.dump(mesh, f)
    draw_matplotlib('test.png', mesh, 500, 500)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'rb') as f:
            mesh = pickle.load(f)
        draw_matplotlib('test.png', mesh, 500, 500)
    else:
        main()
