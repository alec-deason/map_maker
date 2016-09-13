import numpy as np
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter

def main():
    elevation = np.load('elevation.npy')
    water = np.load('water.npy')

    visible_water = water > 300

    water_for_veg = gaussian_filter(water, sigma=1)*2+water
    vegetation = (water_for_veg/water_for_veg[~visible_water].max())*~visible_water
    imsave('combined.png', np.array([elevation*~visible_water, elevation*~visible_water + vegetation*3*elevation.max(), elevation*~visible_water + elevation.max()*visible_water]).T)
    imsave('vegetation.png', vegetation)

if __name__ == '__main__':
    main()
