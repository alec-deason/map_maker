from map_maker.mesh import delauny_mesh
from map_maker.elevation import simplex_noise_elevation
from map_maker.hydrology import hydrolic_erosion
from map_maker.cities import place_seeds
from map_maker.visualization import draw

def main():
    mesh = delauny_mesh(1500, 1500, 10000)
    mesh.elevation = simplex_noise_elevation(mesh)
    #TODO: What's the right interface here? These should be consistent.
    # Either modify the mesh in place or return a modified copy.
    hydrolic_erosion(mesh)
    cities = place_seeds(mesh, 5)
    draw('test.png', mesh, cities)

if __name__ == '__main__':
    main()
