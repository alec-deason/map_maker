import os.path

import numpy as np

from scipy.misc import imsave

def visualize_elevation(ctx, config):
    imsave(os.path.join(config['output_dir'], 'elevation.png'), ctx['elevation'])

def visualize_water(ctx, config):
    water = ctx['water']
    elevation = ctx['elevation']
    visible_water = water > 300

    imsave(os.path.join(config['output_dir'], 'combined.png'), np.array([elevation*~visible_water, elevation*~visible_water, elevation*~visible_water + elevation.max()*visible_water]).T)
    imsave(os.path.join(config['output_dir'], 'water.png'), water)
