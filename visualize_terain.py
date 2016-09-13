import numpy as np
from scipy.misc import imsave

def main():
    elevation = np.load('elevation.npy')
    water = np.load('water.npy')
    
    visible_water = water > 300

    imsave('combined.png', np.array([elevation*~visible_water, elevation*~visible_water, elevation*~visible_water + elevation.max()*visible_water]).T)
    imsave('elevation.png', elevation)
    imsave('water.png', water)

if __name__ == '__main__':
    main()
