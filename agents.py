# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import List, Tuple
import math

import dubins
import rl_gas_survey_dubins_agent_env

def argmax_all(a: np.ndarray) -> np.ndarray:
    """
    Return the indices of every element equal to the global maximum.

    Parameters
    ----------
    a : np.ndarray
        2-D (or N-D) array.

    Returns
    -------
    np.ndarray
        An array of shape (k, a.ndim) where each row is the index of one
        max-valued element.  For 2-D input the columns are (row, col).
    """
    if a.size == 0:
        raise ValueError("Input array is empty.")

    max_val = a.max()               # global maximum
    return np.argwhere(a == max_val)

def plot_n(x, y, data_list, titles=None, x_range=(0, 250), y_range=(0, 250), path=None):
    fig_combined, axes = plt.subplots(1, len(data_list), figsize=(2+5*len(data_list), 5))
    if len(data_list) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, ax in enumerate(axes):
        sc = ax.scatter(x, y,
                    c=data_list[i],
                    cmap="coolwarm",
                    s=1)
        if path is not None:
            ax.scatter(path[:, 0], path[:, 1], c='black', s=1)

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel('Easting [m]')
        if not i:
            ax.set_ylabel('Northing [m]')
        
        if titles is not None:
            ax.set_title(titles[i])
    
        cbar = fig_combined.colorbar(sc, ax=ax)
        cbar.set_label('Value')
    
    plt.show()

def plot_env(env,
             *,                     # force keyword use
             x=None, y=None, c=None,
             path=None,
             x_range=(0, 250), y_range=(0, 250),
             vmin=None, vmax=None,
             cmap="coolwarm",
             ax=None):
    """
    Visualise GP prediction (mean or variance) on an existing axis.

    Parameters
    ----------
    x, y : 1-D tensors or arrays with the same length
    c    : 1-D tensor/array of scalar values (mean, var, etc.)
    path : (N,2) array of visited coordinates
    ax   : matplotlib Axes; if None, a new fig/ax pair is created.
    """
    if x is None:
        x = env._coord_x
    if y is None:
        y = env._coord_y
    if c is None:
        c = env.values                           # default field to show

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 255

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sc = ax.scatter(x, y,
                    c=c,
                    cmap=cmap,
                    s=1,
                    vmin=0,
                    vmax=255)
    if path is not None:
        ax.scatter(path[:, 0], path[:, 1], c='black', s=1)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    # add a colour-bar only if we made the axis ourselves
    if created_ax:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Value')

    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title(f"Time {env.time}, {env.parameter} at {env.depth} m")

    return sc

def compare_envs(envs,
                 *,
                 env_names = ['Env1', 'Env2', 'Env3'],
                 mean_attr  ="pred_mu_norm_clipped",      # attribute names on the env
                 var_attr   ="pred_var_norm_clipped",
                 path = False,    # optional
                 data_min=0, data_max=255,
                 cmap="coolwarm",
                 x_range=(0, 250), y_range=(0, 250),
                 figsize_per_row=(10, 4)):
    """
    Draw a 2-column grid of mean / variance for each environment in `envs`.

    Returns
    -------
    fig, axes : the matplotlib figure and (N,2) array of Axes.
    """
    n = len(envs)
    fig_w = figsize_per_row[0]
    fig_h = figsize_per_row[1] * n
    fig, axes = plt.subplots(nrows=n,
                             ncols=2,
                             figsize=(fig_w, fig_h),
                             constrained_layout=True,
                             sharex=True, sharey=True)

    # If only one environment, axes is 1-D; force 2-D shape for uniformity
    if n == 1:
        axes = axes.reshape(1, 2)

    first_artist = None

    for i, env in enumerate(envs):
        if path:
            env_path = env.sampled_coords[:env.sample_idx]
        else:
            env_path = None
        # Column 0: predicted mean
        artist_mean = plot_env(env,
                    c=getattr(env, mean_attr),
                    path=env_path,
                    x_range=x_range, y_range=y_range,
                    ax=axes[i, 0])

        first_artist = first_artist or artist_mean
        axes[i, 0].set_ylabel(f"{env_names[i]}", rotation=90, labelpad=40)

        # Column 1: predicted variance
        plot_env(env,
                    c=getattr(env, var_attr),
                    path=env_path,
                    x_range=x_range, y_range=y_range,
                    ax=axes[i, 1])

    # Column headers
    axes[0, 0].set_title("Predicted mean")
    axes[0, 1].set_title("Predicted variance")

    # ───────────── shared colour-bar ─────────────
    if first_artist is not None:
        norm = Normalize(vmin=data_min, vmax=data_max)
        sm   = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])                           # dummy for MPL <3.8
        
        cbar = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.05,    # height of the bar  (tweak to taste)
            pad=0.08)         # gap between bar and sub-plots
        cbar.set_label("Value")

    return fig, axes

def in_square(coords: np.ndarray,
              min_x: float, max_x: float,
              min_y: float, max_y: float,
              strict: bool = True,
              return_points: bool = False):
    """
    Check which (x, y) coords lie inside the axis-aligned square/rectangle.

    Parameters
    ----------
    sample_coords : np.ndarray, shape (N, 2)
        Array of [x, y] coordinates.
    min_x, max_x, min_y, max_y : float
        Bounds of the square/rectangle.
    strict : bool
        If True, use strict inequalities (min < x < max, min < y < max).
        If False (default), include the boundary (min <= x <= max ...).
    return_points : bool
        If True, also return the filtered coordinates.

    Returns
    -------
    mask : np.ndarray, shape (N,)
        Boolean mask where True means the point is inside.
    points_inside : np.ndarray, shape (M, 2)
        Only if return_points=True: the points that are inside.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    if strict:
        mask = (x > min_x) & (x < max_x) & (y > min_y) & (y < max_y)
    else:
        mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

    if return_points:
        return mask, coords[mask]
    return mask

def in_circle(coords: np.ndarray,
              cx: float, cy: float, r: float,
              strict: bool = True,
              return_points: bool = False):
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    if strict:
        mask = (cx - x)**2 + (cy - y)**2 < r*r
    else:
        mask = (cx - x)**2 + (cy - y)**2 <= r*r
    
    if return_points:
        return mask, coords[mask]
    return mask

class adaptive_agents():
    def __init__(self, *args, model_type, obs, kappa=None, gamma=None, debug=False, turn_radius=25, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.type = model_type
        if self.type not in ['IG', 'UCB', 'DUCB']:
            print(f'Agent type {self.type} not recognized.')
        
        obs = obs[0]
        
        self._coord_y = obs['map'][2]
        self._coord_x = obs['map'][3]
        self._coords = np.stack([self._coord_x, self._coord_y], axis=-1)

        self.x_max = self._coord_x.max()
        self.y_max = self._coord_y.max()

        self.loc_action_space = obs['loc'] # [-1, 1]
        self.loc = (self.loc_action_space + 1)/2 * [self.x_max, self.y_max]
        self.hdg = obs['hdg'] # onehot e.g. [0, 1, 0, 0]

        self.turn_radius = turn_radius
        self.path_planner = dubins.Dubins(self.turn_radius-3, 1.0) #1.0 - sample every meter
        
        # tuning params
        self.kappa = kappa or 1.0
        self.gamma = gamma or 1.0

        self.debug = debug
    
    def get_wps_to_max_objective(self, obs):
        obs = obs[0]

        self.gas = obs['map'][0]
        self.var = obs['map'][1]
        self.loc_action_space = obs['loc']
        self.loc = (self.loc_action_space + 1)/2 * [self.x_max, self.y_max]
        self.hdg = obs['hdg']
        if self.debug:
            print(f'loc_action_space: {self.loc_action_space}, loc: {self.loc}, hdg: {self.hdg}')

        self.dx = self._coord_x - self.loc[0]
        self.dy = self._coord_y - self.loc[1]
        self.dist = np.hypot(self.dx, self.dy)

        # - Turn radius masks -
        # Find center of circles
        pi = math.pi
        a = pi*self.hdg.argmax()/4
        rc_rot = a - pi/2
        lc_rot = a + pi/2
        rc_centre = (self.loc[0] + math.cos(rc_rot)*self.turn_radius, self.loc[1] + math.sin(rc_rot)*self.turn_radius)
        lc_centre = (self.loc[0] + math.cos(lc_rot)*self.turn_radius, self.loc[1] + math.sin(lc_rot)*self.turn_radius)

        # Compute left and right masks and total turn radius mask
        rc_mask = in_circle(self._coords, rc_centre[0], rc_centre[1], self.turn_radius)
        lc_mask = in_circle(self._coords, lc_centre[0], lc_centre[1], self.turn_radius)
        rc_dist = rc_mask*(self.dist - self.turn_radius*2)
        lc_dist = lc_mask*(self.dist - self.turn_radius*2)
        self.turn_radius_mask = rc_dist + lc_dist

        match self.type:
            case 'IG':
                # Highest entropy reduction, in practice go to location with max variance
                # If multiple locations are tied, go to nearest
                self.map = self.var + self.turn_radius_mask

            case 'UCB':
                # Balances entropy reduction with sampling of high concentrations
                # If multiple locations are tied, go to nearest
                self.gas_scaled = np.clip(self.gas * self.kappa, 0, 255)
                self.map = self.gas_scaled + self.var + self.turn_radius_mask

            case 'DUCB':
                # Balances entropy reduction with sampling of high concentrations
                # and distance
                # If multiple locations are tied, go to nearest
                self.gas_scaled = np.clip(self.gas * self.kappa, 0, 255)
                self.map = self.gas_scaled + self.var + self.dist * self.gamma + self.turn_radius_mask

            case _:
                print(f'{self.type} agent not implemented')
        
        best_value_idx = self._idx_sorted_by_value_then_distance(field=self.map, ascending_value=False)
        if self.debug:
            print(f'Top map indexes: {best_value_idx[:3]}')
            print("chosen max-value locations:", [self._coords[lo] for lo in best_value_idx[:3]],
                "value:", [self.map[lo] for lo in best_value_idx[:3]],
                "distance:", [self.dist[lo] for lo in best_value_idx[:3]])


        found_new_xy = False
        i = 0
        while found_new_xy is False:
            waypoints_per_heading = []
            new_headings = []
            new_xy = self._coords[best_value_idx[i]]
            start = (self.loc[0], self.loc[1], rl_gas_survey_dubins_agent_env.onehot_to_rad(self.hdg))
            
            # For each heading, find a path from start (self.loc, self.hdg) to end (new_xy, new_heading)
            # Skip end positions that are facing the boundary
            for heading in range(len(self.hdg)):
                new_heading = np.zeros(len(self.hdg))
                new_heading[heading] += 1

                end = (new_xy[0], new_xy[1], rl_gas_survey_dubins_agent_env.onehot_to_rad(new_heading))
                if not rl_gas_survey_dubins_agent_env.facing_the_boundary(new_xy, new_heading, 250, 250, self.turn_radius):
                    sample_coords_xy = self.path_planner.dubins_path(start, end)
                    inside_bools = in_square(sample_coords_xy, 0, self.x_max, 0, self.y_max)
                    
                    if inside_bools.sum() == len(sample_coords_xy):
                        waypoints_per_heading.append(sample_coords_xy)
                        new_headings.append(new_heading)
            
            # If any paths are found, sort paths by length, and return the shortest one
            if len(waypoints_per_heading):
                len_waypoints = [len(wp) for wp in waypoints_per_heading]
                wp_idx_by_length = self._get_idx_sorted_by_value(len_waypoints)
                wp_idx_by_length = [idx[0] for idx in wp_idx_by_length]
                found_new_xy = True
                if self.debug:
                    print(f'Possible valid paths were found. len_waypoints: {len_waypoints} ')
                    print(f'Sorted indexes of path by length: {wp_idx_by_length}')
                    #print(f'Shortest path: {waypoints_per_heading[wp_idx_by_length[0]]}')
            else:
                if self.debug:
                    print(f'No valid waypoints found for new_xy={new_xy}')
            
            i += 1
        
        return waypoints_per_heading[wp_idx_by_length[0]], new_headings[wp_idx_by_length[0]]

    def _idx_sorted_by_value_then_distance(self,
        field: np.ndarray,
        ascending_value: bool = True,
        ascending_distance: bool = True,
    ) -> List[Tuple[int, ...]]:
        """ Return indices of `field` sorted by value, breaking ties by `distances`.

        Parameters
        ----------
        field : np.ndarray
            1D or ND array of values to sort by (primary key).
        distances : np.ndarray
            Same shape as `field`; used as secondary key (tie-breaker).
        ascending_value : bool
            Sort values ascending if True, descending if False.
        ascending_distance : bool
            Sort distances ascending if True (closer first), descending if False.

        Returns
        -------
        idx_list : list[tuple[int, ...]]
            Indices into `field` in the requested order. """

        arr = np.asarray(field)
        dist = np.asarray(self.dist)
        if arr.shape != dist.shape:
            raise ValueError(f"Shape mismatch: field {arr.shape} vs distances {dist.shape}")

        v = arr.ravel()
        d = dist.ravel()

        # Keys: last key is primary for lexsort
        vkey = v if ascending_value else -v
        dkey = d if ascending_distance else -d

        order = np.lexsort((dkey, vkey))  # primary: value; secondary: distance
        return [np.unravel_index(i, arr.shape) for i in order]
    
    def _get_idx_sorted_by_value(self, field: np.ndarray, ascending: bool = True):
        """
        Return indices of `field` sorted by its values.

        Parameters
        ----------
        field : np.ndarray
            1D or ND numeric array.
        ascending : bool
            True → smallest first, False → largest first.

        Returns
        -------
        idx_list : list[tuple[int, ...]]
            List of index tuples (i, j, ...) in the requested order.
        """
        arr = np.asarray(field)
        flat = arr.ravel()
        order = np.argsort(flat, kind="stable")
        if self.debug:
            print(f'type(ascending): {type(ascending)} ascending: {ascending}')
        if not ascending:
            order = order[::-1]
        # map flat indices back to ND tuples
        idx_list = [np.unravel_index(i, arr.shape) for i in order]
        return idx_list

    def _get_best_value_idx(self, field):
        # Find idx of field with the highest value.
        # If multiple locations are tied, go to nearest
        
        var_max_indexes = argmax_all(field)
        dists = self.dist[var_max_indexes[:, 0], var_max_indexes[:, 1]]
        best_dist_idx   = dists.argmin()       # index in the max_coords list
        best_value_idx = tuple(var_max_indexes[best_dist_idx])   # (row, col)
        best_value = self.var[best_value_idx]    # the maximal value (for reference)

        if self.debug:
            print("chosen max-value location:", self._coords[best_value_idx],
                "value:", best_value,
                "distance:", dists[best_dist_idx])
        
        return best_value_idx
    


