import torch
import numpy as np
import gpytorch
import math
#import gpt_plot
#import seaborn as sns
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, bounds, resolution, params, debug=False):

        self.bounds_initial = bounds
        self.resolution = int(resolution)
        self.coords_initial = self.create_grid()
        
        self.params = params

        self.values_initial = self.create_scalar_field(debug=debug)
        self.add_environmental_noise()

        self.bounds = bounds#self.get_bounds_for_rotate()
        self.coords = self.coords_initial
        self.values = self.values_initial
        #if self.params['type'] != 'elliptical':
        #    self.coords, self.values = self.rotate_environment()
        #else:
            #self.coords, self.values = self.subset_grid(self.coords_initial)
            #self.coords = self.coords_initial
            #self.values = self.values_initial
        
        self.params['location'] = self.coords[torch.argmax(self.values)]
    
    def add_environmental_noise(self):
        if 'env_noise_mean' in self.params:
            mask = self.values_initial <= self.params['env_noise_mean']
            self.values_initial[mask] += self.params['env_noise_mean']
            if 'env_noise_var' not in self.params:
                noise_var = 0.001
            else:
                noise_var = self.params['env_noise_var']
            self.values_initial = self.values_initial + self.values_initial*torch.randn(len(self.values_initial))*(noise_var**0.5)
            mask = self.values_initial <= 0
            self.values_initial[mask] = 0
            mask = self.values_initial > 1.000
            self.values_initial[mask] = 1.000


    def get_bounds_for_rotate(self):
        x_min = self.bounds_initial[0][0]/math.sqrt(2)
        y_min = self.bounds_initial[1][0]/math.sqrt(2)
        x_max = self.bounds_initial[0][1]/math.sqrt(2)
        y_max = self.bounds_initial[1][1]/math.sqrt(2)
        return [(x_min, x_max), (y_min, y_max)]
        
        
    def create_grid(self):
        bounds = self.bounds_initial

        if len(bounds) == 0:
            bounds = [(-5, 5), (-5, 5)]
        if self.resolution == None:
            self.resolution = 10

        grid_obs = torch.zeros(self.resolution, len(bounds))
        for i in range(len(bounds)):
            grid_obs[:, i] = torch.linspace(bounds[i][0], bounds[i][1], self.resolution)
        
        return gpytorch.utils.grid.create_data_from_grid(grid_obs)

    
    def create_scalar_field(self, debug=False):

        env_type = self.params['type']
        
        if debug:
            print(f'Creating {env_type} scalar field')

        if env_type == 'cosine':
            return self.scalar_field_cosine()
        elif env_type == 'isotropic':
            return self.scalar_field_isotropic()
        elif env_type == 'anisotropic':
            return self.scalar_field_anisotropic()
        elif env_type == 'elliptical':
            return self.scalar_field_elliptical()
        else:
            raise ValueError('Invalid environment type')


    def scalar_field_cosine(self):
        return torch.cos((self.coords[:, 0] + self.coords[:, 1]) * (2 * torch.pi))# + torch.randn_like(train_x[:, 0]).mul(0.01)


    def scalar_field_isotropic(self):
        # calculate the center of the coordinates
        center = torch.mean(self.coords_initial, dim=0)

        # calculate the distance of each coordinate from the center
        distance = torch.norm(self.coords_initial - center, dim=1)

        # create a scalar field that has a maximum at the center and decreases as the distance from the center increases
        return torch.exp(-(distance**2))


    def scalar_field_anisotropic(self):
        
        center = torch.Tensor(self.params['location_initial'])

        # calculate the distance of each coordinate from the center
        distance = torch.norm(self.coords_initial - center, dim=1)

        x_vals = self.coords_initial[:, 0]
        y_vals = self.coords_initial[:, 1]
        
        x_distance = torch.abs(x_vals - center[0].item())
        y_distance = torch.abs(y_vals - center[1].item())

        scalar_field = torch.zeros_like(x_vals)

        # treat the two halves (x-halves) differently - leads to non-smooth transition
        # dilution controls the decay rate in both x and y directions
        # current_strength controls the ratio between anistropy in x and y direction
        c = self.params['current_strength']
        d = self.params['dilution']

        scalar_field[x_vals > center[0].item()] = \
            torch.exp(-( d*(x_distance[x_vals > center[0].item()]**2 + c*y_distance[x_vals > center[0].item()]**2) ))
        
        scalar_field[x_vals <= center[0].item()] = \
            torch.exp(-(distance[x_vals <= center[0].item()]**2)/2)

        return scalar_field

    def scalar_field_elliptical(self):

        center = torch.Tensor(self.params['location_initial'])

        rel_dists = self.coords_initial - center
        distances = torch.norm(rel_dists, dim=1) # scale with dilution
        angles = torch.atan2(rel_dists[:,1], rel_dists[:,0]) # [-pi, pi]
        
        c = self.params['current_strength']
        d = self.params['dilution']
        cdir = self.params['current_direction']

        angles_rot = angles - cdir/180*math.pi #[-pi-cdir, pi-cdir]
        # Normalize to the interval [-pi, pi]
        angles_rot = (angles_rot + math.pi) % (2 * math.pi) - math.pi
        angles_cos = (-torch.cos(angles_rot) + 1)*c
        angles_cos += 1

        scalar_field = torch.exp(-(distances*d*(angles_cos)))

        return scalar_field
    
  
    def rotate_environment(self):
        ''' Takes the original grid in self.coords (of dim grid_len), rotates it
        and snips away all coordinates outside of +- leak_len (grid_len/math.sqrt(2)).
        Also snips self.values accordingly. '''

        debug = False
        grid = self.coords_initial
        alpha = self.params['current_direction']

        # Convert angle to radians
        alpha_rad = torch.tensor(alpha, dtype=torch.float32) * (torch.pi / 180)

        rotation_matrix = torch.tensor([[torch.cos(alpha_rad), -torch.sin(alpha_rad)],
                                        [torch.sin(alpha_rad), torch.cos(alpha_rad)]]).float()
        
        grid_rot = torch.matmul(rotation_matrix, grid.t().float()).t().double()
        
        return self.subset_grid(grid_rot) # This is a tuple (grid_rot[mask], values[mask])


    def subset_grid(self, grid_rot):
        # Create a mask of boolean values representing whether each coordinate is within the bounds

        x_lim = self.bounds_initial[0][1]/math.sqrt(2)
        y_lim = self.bounds_initial[1][1]/math.sqrt(2)

        mask = (grid_rot[:, 0] >= -x_lim) & (grid_rot[:, 0] <= x_lim) & \
            (grid_rot[:, 1] >= -y_lim) & (grid_rot[:, 1] <= y_lim)

        # Index the coords and values tensors using the mask to get the subset of coordinates and values within the bounds
        return (grid_rot[mask], self.values_initial[mask])


def get_random_env(leak_len=40, type='anisotropic', env_params=None, debug=False):
    leak_len = leak_len # minimum dilution 0.0025
    grid_len = int(leak_len*math.sqrt(2))
    xmin = -grid_len; xmax = grid_len; ymin = -grid_len; ymax = grid_len
    grid_bounds = [(xmin, xmax), (ymin, ymax)]
    grid_resolution = grid_len*2+1

    rng = np.random.default_rng()

    if env_params == None:
        env_params = {
            'type': type
        }
    
    if 'type' not in env_params:
        env_params['type'] = type

    # Setup rest of env_params with random numbers if necessary. This means you can call
    # this function with some env_params already set, if you want them to be static.
    if type == 'elliptical':
        c_low = 20 ; c_high = 200 #c_low = 1 ; c_high = 40
        d_low = 400; d_high = 1000
    else:
        c_low = 1; c_high = 20
        d_low = 25; d_high = 1000


    # Current
    if 'current_strength' not in env_params:
        env_params['current_strength'] = rng.integers(low=c_low, high=c_high, endpoint=True) # [1 20]
    if 'current_direction' not in env_params:
        env_params['current_direction'] = rng.integers(low=-180, high=180, endpoint=True)  # [-180 180]

    # Plume
    if 'dilution' not in env_params:
        env_params['dilution'] = rng.integers(low=d_low, high=d_high)/10000.0 
        # [0.0025 0.1], corresponding to leak_lens of [40 5]
        # elliptic: [0.05 0.1]
    if 'location_initial' not in env_params:
        #loc_min = -leak_len/2
        loc_min = -14 #Tries to ensure that two lines always crosses the plume
        loc_max = -loc_min
        env_params['location_initial'] = [rng.integers(low=loc_min, high=loc_max, endpoint=True),
                                  rng.integers(low=loc_min, high=loc_max, endpoint=True)]
    
    if 'env_noise_mean' not in env_params:
        env_params['env_noise_mean'] = 0.02
    if 'env_noise_var' not in env_params:
        env_params['env_noise_var'] = 0.1

    return Environment(grid_bounds, grid_resolution, env_params, debug)