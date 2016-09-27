from map_maker.mesh import delauny_mesh
from map_maker.elevation import simplex_noise_elevation, cones_elevation
from map_maker.hydrology import hydrolic_erosion
from map_maker.cities import place_seeds, grow_population
from map_maker.visualization import draw

def main():
    mesh = delauny_mesh(1000, 1000, 12000)
    print('Mesh generated')
    mesh.elevation = cones_elevation(mesh)
    print('Initial elevation calculated')
    #TODO: What's the right interface here? These should be consistent.
    # Either modify the mesh in place or return a modified copy.
    hydrolic_erosion(mesh)
    print('Erosion finished')
    mesh = place_seeds(mesh, 15)
    print('Cities seeded')
    mesh = grow_population(mesh, 1000000)
    print('Population distributed')
    draw('test.png', mesh)

if __name__ == '__main__':
    main()
