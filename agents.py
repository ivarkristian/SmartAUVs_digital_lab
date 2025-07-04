# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

class adaptive_agents():
    def __init__(self, *args, type, obs, kappa=None, gamma=None, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = type
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

        # tuning params
        self.kappa = kappa or 1.0
        self.gamma = gamma or 1.0

        self.debug = debug
        
    
    def get_action(self, obs):
        obs = obs[0]

        self.gas = obs['map'][0]
        self.var = obs['map'][1]
        self.loc_action_space = obs['loc']
        self.loc = (self.loc_action_space + 1)/2 * [self.x_max, self.y_max]
        if self.debug:
            print(f'loc_action_space: {self.loc_action_space}, loc: {self.loc}')

        self.dx = self._coord_x - self.loc[0]
        self.dy = self._coord_y - self.loc[1]
        self.dist = np.hypot(self.dx, self.dy)

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
                # and distance
                # If multiple locations are tied, go to nearest
                self.gas_scaled = np.clip(self.gas * self.kappa, 0, 255)
                self.map = self.gas_scaled + self.var + self.dist * self.gamma

            case _:
                print(f'{self.type} agent not implemented')
        
        best_value_idx = self._get_best_value_idx(self.map)
        action = (self._coords[best_value_idx]/[self.x_max, self.y_max]) * 2.0 - 1
        if self.debug:
            print(f'Found {self.type} action: {action}')
        
        return action
    
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
    


