import numpy as np
from collections import defaultdict
import pickle
import json
import io
import gizeh
import math
from PIL import ImageFilter, Image, ImageEnhance
import os.path

MAX_ELEVATION = 100000
class Transitions:
    def __init__(self, choices, p):
        self.choices = choices
        self.p = p / np.sum(p)

    def choose(self):
        choice = np.random.choice(self.choices, p=self.p)
        return choice()

class RidgeLine:
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.l = 0
        self.a = a
        self.growth_rate = 3
        self.angle_jitter = 1
        self.active = True
        self.transitions = Transitions(*zip(*[
            (self.grow, 0.5),
            (self.branch, 0.003),
            (self.turn, 0.25),
            (lambda: None, 0.3)]))

    def tick(self):
        if self.active:
            return self.transitions.choose()

    def grow(self):
        self.l += self.growth_rate

    def branch(self):
        x = self.x + math.cos(self.a) * self.l
        y = self.y + math.sin(self.a) * self.l
        
        branches = []

        a = self.a + (np.random.random() - 0.5) * self.angle_jitter
        branches.append(RidgeLine(x, y, a))

        a = self.a + (np.random.random() - 0.5) * self.angle_jitter
        branches.append(RidgeLine(x, y, a))
        
        self.active = False
        return branches
    
    def turn(self):
        x = self.x + math.cos(self.a) * self.l
        y = self.y + math.sin(self.a) * self.l
        a = self.a + (np.random.random() - 0.5) * self.angle_jitter
        self.active = False
        return [RidgeLine(x, y, a)]

    def bounds(self):
        return ((self.x, self.y), (self.x + math.cos(self.a) * self.l, self.y + math.sin(self.a) * self.l))

    def draw(self, surface, scale, x_offset, y_offset):
        x = self.x - x_offset
        x *= scale
        y = self.y - y_offset
        y *= scale
        xx = (self.x + math.cos(self.a) * self.l) - x_offset
        xx *= scale
        yy = (self.y + math.sin(self.a) * self.l) - y_offset
        yy *= scale
        line = gizeh.polyline(points=[(x,y), (xx,yy)], stroke_width=4, stroke=(0,0,0))
        line.draw(surface)
        gizeh.circle(2, xy=(x,y), fill=(0,0,0)).draw(surface)
        gizeh.circle(2, xy=(xx,yy), fill=(0,0,0)).draw(surface)


from scipy.ndimage.filters import maximum_filter, gaussian_filter, median_filter
from scipy.ndimage import grey_dilation
from scipy.misc import imsave
from scipy.ndimage import prewitt
def make_elevations_numpy(im):
    elevations = np.array(im)[:,:, 3].astype(float) * MAX_ELEVATION
    elevations *= MAX_ELEVATION/elevations.max()
    for i in range(200):
        elevations += gaussian_filter(grey_dilation(elevations, (3,3)), sigma=4)*0.5
        elevations += np.random.randint(MAX_ELEVATION*0.005, size=(500,500))
        elevations *= MAX_ELEVATION/elevations.max()
    return elevations


def make_ridges():
    ridges = []
    for i in range(3):
        ridges.append(RidgeLine(np.random.randint(100), np.random.randint(100), np.random.random()*2*math.pi))
    active_ridges = ridges
    for i in range(300):
        print(i)
        new_active_ridges = []
        for r in ridges:
            result = r.tick()
            if result:
                new_active_ridges.extend(result)
                ridges.extend(result)
            else:
                new_active_ridges.append(r)
        active_ridges = new_active_ridges
    surface = gizeh.Surface(width=500, height=500, bg_color=(0,0,0,0))
    xs, ys = zip(*[p for r in ridges for p in r.bounds()])
    x_offset = np.min(xs)
    y_offset = np.min(ys)
    w = np.max(xs) - x_offset
    h = np.max(ys) - y_offset
    scale = min(w/500, h/500)
    for r in ridges:
        r.draw(surface, scale, x_offset, y_offset)
    data = io.BytesIO()
    surface.write_to_png(data)
    img = Image.open(data)
    return img

def fast_water(elevations, prefix):
    steepness, next_point_x, next_point_y = recalculate_slopes(elevations)
    imsave(prefix + '_steepness.png', steepness)
    imsave(prefix + '_elevation.png', elevations)

    drops = 10000

    flow = np.zeros((500,500))
    carrying = np.zeros(drops)
    deposits = np.zeros((500,500))
    xs = np.random.randint(500, size=drops)
    ys = np.random.randint(500, size=drops)
    next_points_x = next_point_x[xs, ys]
    next_points_y = next_point_y[xs, ys]
    f = np.logical_and((next_points_x != xs), (next_points_y != ys))
    next_points_x = next_points_x[f]
    next_points_y = next_points_y[f]
    carrying = carrying[f]
    i = 0
    min_steepness = np.min(steepness)
    max_steepness = np.max(steepness)
    steepness_scale = max_steepness - min_steepness
    while len(next_points_x) > 0 and i < 100:
        i += 1
        flow[next_points_x, next_points_y] += 1
        #pickup = np.minimum(0, np.arctan(((steepness[next_points_x, next_points_y] + min_steepness) / steepness_scale) * 8 + 4))
        pickup = ((steepness[next_points_x, next_points_y] + min_steepness) / steepness_scale) > 0.01
        import pdb; pdb.set_trace()
        pickup = pickup.astype(int)
        carrying = np.minimum(carrying + pickup, 0)
        to_deposit = -np.minimum(carrying, pickup)
        
        deposits[next_points_x, next_points_y] += to_deposit

        new_next_points_x = next_point_x[next_points_x, next_points_y]
        new_next_points_y = next_point_y[next_points_x, next_points_y]
        f =np.logical_and((new_next_points_x != next_points_x), (new_next_points_y != next_points_y))
        next_points_x = new_next_points_x[f]
        next_points_y = new_next_points_y[f]
        carrying = carrying[f]

    elevations += gaussian_filter(deposits*10, sigma=1)
    elevations = np.clip(elevations, 0, MAX_ELEVATION)

    imsave(prefix + '_flow.png', flow!=0)

    return elevations

def run_fast_waters(elevations):
    for iteration in range(100):
        print('iteration {}'.format(iteration))
        elevations = fast_water(elevations, os.path.join('data/{}'.format(iteration)))

def overlay_water(elevations):
    water = np.zeros(elevations.shape)
    for iteration in range(100):
        if iteration%10 ==0:
            print(iteration)
        water += 1
        mean = median_filter(elevations + water, 3)
        slope = (elevations + water) - mean
        new_water = water - np.maximum(water, slope)
        new_water += np.minimum(np.minimum(slope, median_filter(water, footprint=[[True,True,True],[True,False,True],[True,True,True]])), 0)
        flow = np.abs(water-new_water)
        water = new_water
        imsave('data/elevation_{}.png'.format(iteration), elevations)
        imsave('data/water_{:03d}.png'.format(iteration), water)
        imsave('data/flow_{:03d}.png'.format(iteration), flow)

def recalculate_slopes(elevations):
    width, height = elevations.shape

    axis = np.arange(width).repeat(height).reshape(width,height)
    seed = np.random.randint(1000000)
    r = np.random.RandomState(seed)
    neighboors_x = r.permutation(np.array([axis.T+xx for xx in range(-1,2) for yy in range(-1,2)]))
    r = np.random.RandomState(seed)
    neighboors_y = r.permutation(np.array([axis+yy for xx in range(-1,2) for yy in range(-1,2)]))

    neighboors = np.pad(elevations, 1, 'constant', constant_values=[np.max(elevations)+1])[np.pad(neighboors_x+1, 1, 'constant', constant_values=[1])[1:-1,:,:], np.pad(neighboors_y+1, 1, 'constant', constant_values=[1])[1:-1,:,:]]
    neighboors = neighboors[:,1:-1,1:-1]

    least_neighbor = np.argmin(neighboors, axis=0)

    next_point_x = np.choose(least_neighbor, neighboors_x)
    next_point_y = np.choose(least_neighbor, neighboors_y)
    steepness = elevations - elevations[next_point_x, next_point_y]
    return steepness, next_point_x, next_point_y

if __name__ == '__main__':
    data_path = 'data'
    if not os.path.exists(os.path.join(data_path, 'ridges.png')):
        print('Making ridges')
        img = make_ridges()
        img.save(os.path.join(data_path, 'ridges.png'))
    else:
        print('Loading ridges')
        img = Image.open(os.path.join(data_path, 'ridges.png'))

    if not os.path.exists(os.path.join(data_path, 'initial_elevations.pickle')):
        print('Making initial elevation')
        elevations = make_elevations_numpy(img)
        imsave(os.path.join(data_path, 'initial_elevations.png'), elevations)
        with open(os.path.join(data_path, 'initial_elevations.pickle'), 'wb') as f:
            pickle.dump(elevations, f)
    else:
        print('Loading initial elevation')
        with open(os.path.join(data_path, 'initial_elevations.pickle'), 'rb') as f:
            elevations = pickle.load(f)

    print('Running water')
    #terrain = run_fast_waters(elevations)
    overlay_water(elevations)

