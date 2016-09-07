import numpy as np
import scipy
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


from scipy.ndimage.filters import maximum_filter, gaussian_filter, median_filter, minimum_filter
from scipy.ndimage import grey_dilation, convolve
from scipy.misc import imsave
from scipy.ndimage import prewitt
def make_elevations_numpy(im):
    elevations = np.array(im)[:,:, 3].astype(float) * MAX_ELEVATION
    elevations *= MAX_ELEVATION/elevations.max()
    for i in range(200):
        elevations += gaussian_filter(grey_dilation(elevations, (3,3)), sigma=3)*0.5
        elevations += np.random.randint(MAX_ELEVATION*0.005, size=(500,500))
        elevations *= MAX_ELEVATION/elevations.max()
    thresh = np.percentile(elevations, 10)
    elevations[elevations < thresh] = thresh
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


def gradient_water(elevation):
    width, height = elevation.shape
    water = np.zeros(elevation.shape, dtype=np.float64)
    carrying = np.zeros(elevation.shape, dtype=np.float64)
    rain_rate = 5
    water += 2000
    initial_water = water.sum()
    for iteration in range(1000):
        if iteration%10 ==0:
            print(iteration)
        water += np.random.randint(0, rain_rate, size=(width, height))
        water *= initial_water / water.sum()

        gradient = np.gradient(np.pad((elevation+water), 20, 'constant', constant_values=[MAX_ELEVATION*100]), 10)
        gradient = (-gradient[0][20:-20,20:-20], -gradient[1][20:-20,20:-20])
        points_y, points_x = np.mgrid[0:width, 0:height]
        next_point_x = np.array(points_x)
        next_point_x[gradient[1] >= 0] += 1
        next_point_x[gradient[1] < 0] -= 1
        next_point_x = np.clip(next_point_x, 0, width-1)

        next_point_y = np.array(points_y)
        next_point_y[gradient[0] >= 0] += 1
        next_point_y[gradient[0] < 0] -= 1
        next_point_y = np.clip(next_point_y, 0, height-1)

        new_water = np.array(water)
        new_carrying = np.zeros(carrying.shape)
        slope = (elevation + water) - (elevation[next_point_y, next_point_x]+water[next_point_y,next_point_x])
        out_flow = np.maximum(np.minimum(water, slope), 0)
        np.add.at(new_carrying, [next_point_y, next_point_x], carrying)
        carrying = new_carrying

        np.add.at(new_water, [next_point_y, next_point_x], out_flow)
        new_water -= out_flow

        drop = gaussian_filter(new_carrying * (1-out_flow/out_flow.max()), sigma=10)
        elevation += drop
        carrying -= drop
        pickup = gaussian_filter(out_flow * np.minimum(slope, 1000) * 0.01 * (water < 300), sigma=1)
        elevation -= pickup
        carrying += pickup
        elevation = np.clip(elevation, 0, MAX_ELEVATION)

        water = new_water



        if iteration%1==0:
            water_thresh = 100
            covered = water > water_thresh
            covered = covered[2:-2,2:-2]
            base=(elevation[2:-2,2:-2])*~covered
            colored = np.array([base, base, base+(MAX_ELEVATION*covered)])
            scipy.misc.toimage(colored).save('data/water_{:03d}.png'.format(iteration))
            #imsave('data/just_water_{:03d}.png'.format(iteration), water)
            scipy.misc.toimage(elevation, cmin=0, cmax=MAX_ELEVATION).save('data/elevation_{:03d}.png'.format(iteration))

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
    #overlay_water(elevations)
    gradient_water(elevations)

