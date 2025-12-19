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
        if self.type not in ['IG', 'UCB', 'DUCB', 'DUCB_beta']:
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
        if kappa is None:
            self.kappa = 1.0
        else:
            self.kappa = kappa
        
        if gamma is None:
            self.gamma = -1.0
        else:
            self.gamma = gamma

        self.debug = debug
    
    def get_wps_to_max_objective(self, obs, n_samples=0):
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
        #pi = math.pi
        #a = pi*self.hdg.argmax()/4
        #rc_rot = a - pi/2
        #lc_rot = a + pi/2
        #rc_centre = (self.loc[0] + math.cos(rc_rot)*self.turn_radius, self.loc[1] + math.sin(rc_rot)*self.turn_radius)
        #lc_centre = (self.loc[0] + math.cos(lc_rot)*self.turn_radius, self.loc[1] + math.sin(lc_rot)*self.turn_radius)

        # Compute left and right masks and total turn radius mask
        #rc_mask = in_circle(self._coords, rc_centre[0], rc_centre[1], self.turn_radius)
        #lc_mask = in_circle(self._coords, lc_centre[0], lc_centre[1], self.turn_radius)
        #rc_dist = rc_mask*(self.dist - self.turn_radius*2)
        #lc_dist = lc_mask*(self.dist - self.turn_radius*2)
        #self.turn_radius_mask = rc_dist + lc_dist

        self.dubins_dist = dubins_arc_tangent_distance_lr(self.loc, self.hdg, self.turn_radius, self._coords)

        match self.type:
            case 'IG':
                # Highest entropy reduction, in practice go to location with max variance
                # If multiple locations are tied, go to nearest
                self.map = self.var

            case 'UCB':
                # Balances entropy reduction with sampling of high concentrations
                # If multiple locations are tied, go to nearest
                self.gas_scaled = np.clip(self.gas * self.kappa, 0, 255)
                self.map = self.gas_scaled + self.var

            case 'DUCB':
                # Balances entropy reduction with sampling of high concentrations
                # and Dubins distance
                # If multiple locations are tied, go to nearest
                self.var_scaled = self.var * self.kappa
                self.dist_scaled = self.dubins_dist * self.gamma
                self.map = self.gas + self.var_scaled + self.dist_scaled
            
            case 'DUCB_beta':
                # Balances entropy reduction with sampling of high concentrations
                # and distance, evolves beta parameter (kappa)
                # If multiple locations are tied, go to nearest
                # UCB_t(x)= u{t-1}(x) + sqrt{beta_t} * sigma_{t-1}(x)

                # beta_t = 2*log(|D|*pi^2*t^2/(6*d))
                # |D| = number of candidate points you consider
	            # t = iteration (number of samples taken so far)
	            # d ~ (0,1) = failure probability (e.g. 0.1, 0.05)
                D = self._coord_x.shape[-1]*self._coord_y.shape[-1]
                self.beta_t = 2*math.log(D*math.pi**2*n_samples**2/(6*self.kappa))
 
                self.var_scaled = self.var * np.sqrt(self.beta_t)
                self.dist_scaled = self.dubins_dist * self.gamma
                self.map = self.gas + self.var_scaled + self.dist_scaled

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
    

def propagate_rightward_inside_mask(
    dist: np.ndarray,
    mask: np.ndarray,
    *,
    dx: float = 1.0,
    keep_min: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Treat a masked region specially by propagating distances from the left edge
    of each masked segment rightwards, adding propagation distance inside mask.

    For each row y and each contiguous masked segment [xL..xR], define:
        base = dist[y, xL]
        new[y, x] = base + (x - xL)*dx    for x in [xL..xR]

    Parameters
    ----------
    dist : (H, W) array
        Existing scalar distance field.
    mask : (H, W) bool array
        True where special region (e.g., circle) is.
    dx : float
        Physical spacing per pixel in x-direction.
    keep_min : bool
        If True, inside-mask values become min(original, propagated).
        If False, inside-mask values are overwritten by propagated.
    fill_value : float
        If dist has invalid values at the left edge (NaN/inf), this is used
        as base to avoid crashing; propagated values then become fill_value too.

    Returns
    -------
    out : (H, W) array
        Updated distance field.
    """
    dist = np.asarray(dist)
    mask = np.asarray(mask, dtype=bool)
    if dist.ndim != 2 or mask.ndim != 2 or dist.shape != mask.shape:
        print(f'dist: {dist.shape} mask: {mask.shape}')
        raise ValueError("dist and mask must be 2D arrays of the same shape")

    H, W = dist.shape
    out = dist.copy()

    # Process each row independently (perfect for a circular region)
    for y in range(H):
        row_mask = mask[y]
        if not row_mask.any():
            continue

        xs = np.flatnonzero(row_mask)

        # Split into contiguous segments (handles general masks too)
        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends   = np.r_[breaks, len(xs) - 1]

        for si, ei in zip(starts, ends):
            xL = xs[si]
            xR = xs[ei]

            base = dist[y, xL]
            if not np.isfinite(base):
                base = fill_value

            # Rightward propagation within this segment
            ramp = base + (np.arange(xL, xR + 1) - xL) * dx

            if keep_min:
                out[y, xL:xR + 1] = np.minimum(out[y, xL:xR + 1], ramp)
            else:
                out[y, xL:xR + 1] = ramp

    return out

def _dubins_arc_tangent_one_circle(C, R, A, P_flat, original_shape, forward_ccw: bool):
    """
    Internal helper: compute arc+tangent+straight distance from A to every P
    using a single circle (center C, radius R) and a constrained arc direction.

    Parameters
    ----------
    C : (2,) array
    R : float
    A : (2,) array
    P_flat : (N,2) array of targets
    forward_ccw : bool
        True  -> arc along circle is CCW from A
        False -> arc along circle is CW from A

    Returns
    -------
    L_opt : (N,) array of distances
    """
    C = np.asarray(C, dtype=float)
    A = np.asarray(A, dtype=float)
    P_flat = np.asarray(P_flat, dtype=float)

    # Vectors from center to targets
    v = P_flat - C         # (N,2)
    d = np.linalg.norm(v, axis=1)  # (N,)

    # Ensure targets are outside or just treat inside as outside
    mask_inside = d <= R
    if np.any(mask_inside):
        d[mask_inside] = R + 1e-6

    # Angles from center to points
    alpha = np.arctan2(v[:, 1], v[:, 0])  # (N,)
    beta  = np.arccos(R / d)              # (N,)

    # Two tangent angles
    theta1 = alpha + beta
    theta2 = alpha - beta

    # Tangent points
    T1 = C + R * np.column_stack([np.cos(theta1), np.sin(theta1)])  # (N,2)
    T2 = C + R * np.column_stack([np.cos(theta2), np.sin(theta2)])  # (N,2)

    # Starting angle on circle
    theta0 = np.arctan2(A[1] - C[1], A[0] - C[0])

    if forward_ccw:
        # Move CCW from theta0 to the target angle
        dtheta1 = (theta1 - theta0) % (2.0 * np.pi)
        dtheta2 = (theta2 - theta0) % (2.0 * np.pi)
    else:
        # Move CW from theta0 to the target angle
        dtheta1 = (theta0 - theta1) % (2.0 * np.pi)
        dtheta2 = (theta0 - theta2) % (2.0 * np.pi)

    # Arc lengths
    s1 = R * dtheta1
    s2 = R * dtheta2

    # Straight segments
    l1 = np.linalg.norm(P_flat - T1, axis=1)
    l2 = np.linalg.norm(P_flat - T2, axis=1)

    # Total path lengths
    L1 = s1 + l1
    L2 = s2 + l2

    # Take min over the two tangent solutions
    #L_opt = np.minimum(L1, L2)
    if forward_ccw:
    #    L2[mask_inside] = np.inf
    #    return L2
        L2_prop = propagate_rightward_inside_mask(L2.reshape(original_shape), mask_inside.reshape(original_shape))
        return L2_prop
    
    #L1[mask_inside] = np.inf
    #return L1
    L1_prop = propagate_rightward_inside_mask(L1.reshape(original_shape), mask_inside.reshape(original_shape))
    return L1_prop
    
    
def dubins_arc_tangent_distance_lr(
    A,                # agent location (x,y)
    heading_onehot,   # one-hot heading (len=4 or 8)
    R,                # turn radius
    P,                # grid of targets, shape (H,W,2) or (N,2)
    ):
    """
    Compute an approximate Dubins-like distance field from agent pose to each P,
    using left and right turning circles and forward-only arcs.

    Path model for each circle:
        arc along circle (in allowed direction) + straight tangent to P

    Then:
        distance = min(distance_via_left_circle, distance_via_right_circle)

    Parameters
    ----------
    A : array-like shape (2,)
        Agent location.
    heading_onehot : array-like shape (4,) or (8,)
        One-hot heading; index of '1' -> heading angle 0, 90, 180,... or 45,90,...
    R : float
        Turn radius.
    P : ndarray
        Target locations, shape (H,W,2) or (N,2).

    Returns
    -------
    dist : ndarray
        Distance field, same shape as P[...,0].
    """

    A = np.asarray(A, dtype=float)
    P = np.asarray(P, dtype=float)
    heading_onehot = np.asarray(heading_onehot, dtype=float)

    # Flatten P
    original_shape = P.shape[:-1]
    P_flat = P.reshape(-1, 2)

    # --- Heading angle from one-hot ---
    idx = int(np.argmax(heading_onehot))
    n_dirs = len(heading_onehot)
    heading_angle = idx * (2.0 * np.pi / n_dirs)
    h = np.array([np.cos(heading_angle), np.sin(heading_angle)], dtype=float)

    # --- Compute left/right circle centers from agent location and heading ---
    # Left normal = rotate h by +90°, right normal = -left normal
    n_left  = np.array([-h[1], h[0]], dtype=float)
    n_right = -n_left

    C_left  = A + R * n_left
    C_right = A + R * n_right

    # --- Distances via left circle (CCW) and right circle (CW) ---
    L_left  = _dubins_arc_tangent_one_circle(C_left,  R, A, P_flat, original_shape, forward_ccw=True)
    L_right = _dubins_arc_tangent_one_circle(C_right, R, A, P_flat, original_shape, forward_ccw=False)

    L_left = L_left.reshape(original_shape)
    L_right = L_right.reshape(original_shape)
    # --- Take elementwise minimum ---
    L_opt = np.minimum(L_left, L_right)

    return L_opt
    #return L_opt.reshape(original_shape), L_left, L_right

def plot_dubins_distance_contours(
    X, Y, dubins_dist, levels=15, cmap=None,
    euclid_dist=None,
    title="Dubins-based distance field",
    draw_agent=None, save_path=None
):
    """
    Visualize the Dubins distance field with optional agent drawing.

    Parameters
    ----------
    X, Y : 2D arrays
        Grid coordinates.
    dubins_dist : 2D array
        Dubins-like distance field.
    euclid_dist : 2D array, optional
        If provided, adds inset comparison.
    draw_agent : dict, optional
        {
            "loc": (x, y),
            "hdg": one-hot heading vector of length 4 or 8
        }
    """
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    # Filled contour field
    if cmap:
        cs = ax.contourf(X, Y, dubins_dist, levels=levels, cmap=cmap)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Dubins distance")

    # Line contours
    #ax.contour(X, Y, dubins_dist, levels=levels, colors="k", linewidths=0.5)

    # ----------------------------------------------------------------------
    # Draw agent (optional)
    # ----------------------------------------------------------------------
    if draw_agent is not None:
        A = np.asarray(draw_agent["loc"], dtype=float)
        hdg_onehot = np.asarray(draw_agent["hdg"], dtype=float)

        idx = int(np.argmax(hdg_onehot))
        n_dirs = len(hdg_onehot)
        heading_angle = idx * (2.0 * np.pi / n_dirs)

        # Heading unit vector
        h = np.array([np.cos(heading_angle), np.sin(heading_angle)])

        # Arrow length is ~5% of domain size
        domain_scale = 0.08 * max(X.max() - X.min(), Y.max() - Y.min())
        arrow = domain_scale * h

        arrow_color = "black"

        ax.arrow(
            A[0], A[1],
            arrow[0], arrow[1],
            width=domain_scale * 0.03,
            head_width=domain_scale * 0.15,
            head_length=domain_scale * 0.20,
            color=arrow_color,
            length_includes_head=True,
            zorder=5
        )
        ax.plot(A[0], A[1], "ko", markersize=5, zorder=6)
        ax.text(A[0]-domain_scale*0.9, A[1]-domain_scale*0.6, " agent", color=arrow_color, fontsize=10)

    # ----------------------------------------------------------------------
    # Optional inset for Euclidean distance
    # ----------------------------------------------------------------------
    if euclid_dist is not None:
        axins = ax.inset_axes([0.65, 0.05, 0.33, 0.33])
        cs2 = axins.contourf(X, Y, euclid_dist, levels=30, cmap="plasma")
        axins.set_title("Euclidean")
        axins.set_xticks([])
        axins.set_yticks([])
        fig.colorbar(cs2, ax=axins, fraction=0.046)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", dpi=300)

    plt.show()