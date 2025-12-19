import math
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from gpytorch.utils.errors import NotPSDError
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from typing import List

class RotatedMaternARD(gpytorch.kernels.Kernel):
    """
    Rotated Matérn kernel with ARD lengthscales in 2D.

    - Inputs are in the original (x, y) frame.
    - Internally, we rotate by angle_deg into an anisotropy-aligned frame.
    - ARD lengthscales [ell_par, ell_perp] are applied in that rotated frame.
    - nu controls the Matérn smoothness: 0.5, 1.5, or 2.5 supported.
    """
    is_stationary = True
    has_lengthscale = True

    def __init__(self, angle_deg: float = 0.0, ard_num_dims: int = 2, nu: float = 1.5):
        super().__init__(ard_num_dims=ard_num_dims)
        assert ard_num_dims == 2, "This implementation expects 2D inputs."
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError(f"nu must be one of (0.5, 1.5, 2.5), got {nu}")
        self.nu = float(nu)
        self.register_buffer("R", self._make_rotation(angle_deg))

    @staticmethod
    def _make_rotation(angle_deg):
        th = math.radians(angle_deg)
        R = torch.tensor([[ math.cos(th),  math.sin(th)],
                          [-math.sin(th), math.cos(th)]], dtype=torch.float)
        return R

    def set_angle(self, angle_deg: float):
        self.R = self._make_rotation(angle_deg).to(self.R.device)

    def forward(self, x1, x2, diag: bool = False, **params):
        # 1) Rotate into anisotropy-aligned frame
        x1r = x1 @ self.R.T   # (..., 2)
        x2r = x2 @ self.R.T   # (..., 2)

        # 2) Apply ARD lengthscales manually
        # self.lengthscale has shape (1, 2) for ard_num_dims=2
        ls = self.lengthscale  # (1, 2) or (batch, 1, 2)
        while ls.dim() < x1r.dim():
            ls = ls.unsqueeze(-2)  # broadcast to (..., 1, 2)

        x1s = x1r / ls
        x2s = x2r / ls

        # 3) Compute Euclidean distance in scaled, rotated space
        if diag:
            diff = x1s - x2s
            d = diff.pow(2).sum(dim=-1).sqrt()    # (...,)
        else:
            d = torch.cdist(x1s, x2s, p=2)        # (N1, N2)

        # 4) Matérn kernel depending on nu
        if self.nu == 0.5:
            # Matérn 1/2: k(d) = exp(-d)
            k = torch.exp(-d)
        elif self.nu == 1.5:
            # Matérn 3/2: k(d) = (1 + sqrt(3) d) exp(-sqrt(3) d)
            c = math.sqrt(3.0)
            cd = c * d
            k = (1.0 + cd) * torch.exp(-cd)
        elif self.nu == 2.5:
            # Matérn 5/2: k(d) = (1 + sqrt(5) d + 5 d^2 / 3) exp(-sqrt(5) d)
            c = math.sqrt(5.0)
            cd = c * d
            k = (1.0 + cd + (cd ** 2) / 3.0) * torch.exp(-cd)
        else:
            # Should not happen because we validate nu in __init__
            raise RuntimeError(f"Unsupported nu={self.nu}")

        return k

class ExactGP_RotMatARD(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 angle_deg: float,
                 ell_par: float, ell_perp: float,
                 outputscale: float = 1.0,
                 mean_value: float = 0.0,
                 nu: float = 1.5):
        super().__init__(train_x, train_y, likelihood)

        # Constant mean
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant = torch.tensor(
            mean_value,
            dtype=torch.float,
            device=train_x.device if train_x.numel() > 0 else likelihood.noise.device,
        )

        # Rotated Matérn ARD kernel with chosen nu
        base = RotatedMaternARD(angle_deg=angle_deg, ard_num_dims=2, nu=nu)
        ls = torch.tensor([ell_par, ell_perp], dtype=torch.float).view(1, 2)
        base.lengthscale = ls  # ARD vector [ℓ_parallel, ℓ_perp]

        self.covar_module = gpytorch.kernels.ScaleKernel(base)
        self.covar_module.outputscale = torch.as_tensor(outputscale, dtype=torch.float)

    def set_hyperparams(self, ell_par, ell_perp, angle_deg=None, outputscale=None, mean_value=None, noise=None, freeze: bool = True):
        base = self.covar_module.base_kernel  # RotatedMaternARD

        if angle_deg is not None:
            base.set_angle(angle_deg)

        ls = torch.tensor([ell_par, ell_perp], dtype=torch.float).view(1, 2).to(base.lengthscale.device)
        base.lengthscale = ls

        if outputscale is not None:
            self.covar_module.outputscale = torch.as_tensor(outputscale, dtype=torch.float).to(ls.device)
        if mean_value is not None:
            self.mean_module.constant = torch.tensor(mean_value, dtype=torch.float, device=ls.device)
        if noise is not None:
            self.likelihood.noise = torch.as_tensor(noise, dtype=torch.float).to(ls.device)

        if freeze:
            base.raw_lengthscale.requires_grad_(False)
            self.covar_module.raw_outputscale.requires_grad_(False)
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
            self.mean_module.constant.requires_grad_(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

@torch.no_grad()
def build_base_model(angle_deg, ell_par, ell_perp, outputscale, mean_value: float = 0.0,
                     noise: float = 1e-3, nu: float = 1.5, device: str = "cpu"):
    # empty base set (so each subset stands alone)
    X0 = torch.empty(0, 2, dtype=torch.float, device=device)
    y0 = torch.empty(0, dtype=torch.float, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGP_RotMatARD(
        X0, y0, likelihood,
        angle_deg=angle_deg,
        ell_par=ell_par,
        ell_perp=ell_perp,
        outputscale=outputscale,
        mean_value=mean_value,
        nu=nu,
    ).to(device)

    model.set_hyperparams(
        ell_par, ell_perp,
        angle_deg=angle_deg,
        outputscale=outputscale,
        mean_value=mean_value,
        noise=noise,
        freeze=True,
    )
    model.eval()
    likelihood.eval()
    return model, likelihood


@torch.no_grad()
def score_subset_with_swap(model, likelihood, Xk, yk, Xtest, ytrue):
    model.set_train_data(inputs=Xk, targets=yk, strict=False)
    model.eval(); likelihood.eval()

    try:
        with gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(1e-2):
            pred = likelihood(model(Xtest))
    except NotPSDError:
        # Fallback: you decide what makes sense here.
        # For a search over subsets, you might return "bad" scores so this subset is discarded.
        rmse = float("inf")
        nll = float("inf")
        mu = torch.full_like(ytrue, float("nan"))
        var = torch.full_like(ytrue, float("nan"))
        return rmse, nll, mu, var

    mu, var = pred.mean, pred.variance
    rmse = torch.sqrt(torch.mean((mu - ytrue)**2)).item()
    #nll = -pred.log_prob(ytrue).mean().item()
    nll= 0
    return rmse, nll, mu, var

def init_strategy_results(strategy_names):
    """
    Create a results dict for multiple strategies.
    Example keys: ["Lawnmower", "DUCB", "RL"]
    """
    return {
        name: {
            "rmse": [],
            "nll": []
        } for name in strategy_names
    }

@torch.no_grad()
def evaluate_strategies_for_field_lognorm_gridspec(
    base_model, likelihood,
    Xtest, ytrue,
    strategy_samples,
    results,
    threshold=550.0,
    obs_x=None, obs_y=None,
    make_plot=False,
    title_prefix="GP prediction"
):
    """
    Evaluate multiple sampling strategies on ONE true field, and optionally
    create a 2×2 plot of the truth and GP predictions.

    Parameters
    ----------
    base_model : ExactGP_RotMat32ARD
        Base GP model with fixed hyperparams (angle, ell_par, ell_perp, etc.).
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    Xtest : torch.Tensor, shape (N_test, 2)
        Test coordinates (original coordinate frame).
    ytrue : torch.Tensor, shape (N_test,)
        True scalar field values at Xtest.
    strategy_samples : dict
        Mapping from strategy name -> (Xk, yk),
        where Xk, yk are torch tensors for that field.
        Example:
            {
              "Lawnmower": (X_lawn, y_lawn),
              "DUCB":      (X_ducb,  y_ducb),
              "RL":        (X_rl,    y_rl),
            }
    results : dict
        Results dict from init_strategy_results; updated in-place.
    obs_x, obs_y : int, optional
        Grid resolution of Xtest (so that N_test = obs_x * obs_y).
        Required if make_plot=True.
    make_plot : bool
        If True, also generate a 2×2 figure:
        [True field, Lawn­mower pred, DUCB pred, RL pred].
    title_prefix : str
        Prefix for the figure suptitle.

    Returns
    -------
    results : dict
        Updated results with appended RMSE and NLL for this field.
    """

    # Store predictions for plotting if needed
    preds_mu = {}

    for name, (Xk, yk) in strategy_samples.items():
        rmse, nll, mu, var = score_subset_with_swap(
            base_model, likelihood, Xk, yk, Xtest, ytrue
        )
        results[name]["rmse"].append(rmse)
        results[name]["nll"].append(nll)
        preds_mu[name] = mu.detach().cpu().numpy()

    # ----------------------------------------------------------------------
    # Optional plotting: 2×2 (truth + 3 strategy predictions)
    # ----------------------------------------------------------------------
    if make_plot:
        if obs_x is None or obs_y is None:
            raise ValueError("obs_x and obs_y must be provided when make_plot=True")

        # Convert truth and coords to numpy and reshape
        Xtest_np = Xtest.detach().cpu().numpy()
        ytrue_np = ytrue.detach().cpu().numpy()

        X = Xtest_np[:, 0].reshape(obs_y, obs_x)
        Y = Xtest_np[:, 1].reshape(obs_y, obs_x)
        Z_true = ytrue_np.reshape(obs_y, obs_x)

        # Get predictions in a fixed order (assuming keys exist)
        order = ["Lawnmower", "DUCB", "RL"]
        Z_preds = {name: preds_mu[name].reshape(obs_y, obs_x) for name in order}

        fig = plt.figure(figsize=(12, 9), dpi=300)

        # GridSpec: 2 rows, 2 columns, plus margin on right for colorbar
        gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            wspace=-0.28,   # horizontal spacing between plots
            hspace=0.25,   # vertical spacing between plots
            left=0.05,
            right=0.92,    # leave space for colorbar
            bottom=0.07,
            top=0.90
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        for ax in (ax1, ax2, ax3, ax4):
            ax.grid(False)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(axis='both', labelsize=12)
        
        bg = threshold   # your background concentration (or median/mean)
        eps = 1e-2   # small offset to avoid log(0)

        vmin = eps
        vmax = 2000-bg+eps #np.max(excess)

        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = "viridis"
        interp = 'bicubic'

        # --- 1) TRUE FIELD ---
        # Shift field so that bg → ≈0, then clip
        excess_true = np.clip(Z_true - bg + eps, eps, None)
        im1 = ax1.imshow(
            excess_true,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap=cmap,
            norm=norm,
            interpolation=interp
        )
        ax1.set_title("GP prediction - True field", fontsize=14)
        ax1.set_aspect("equal"); ax1.set_xlabel("East [m]"); ax1.set_ylabel("North [m]")

        # --- 2) LAWN ---
        ex2 = np.clip(Z_preds['Lawnmower'] - bg + eps, eps, None)
        im2 = ax2.imshow(ex2, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap=cmap, norm=norm, interpolation=interp)
        ax2.set_title("GP prediction - lawnmower", fontsize=14)
        ax2.set_aspect("equal"); ax2.set_xlabel("East [m]"); ax2.set_ylabel("North [m]")

        # --- 3) DUCB ---
        ex3 = np.clip(Z_preds["DUCB"] - bg + eps, eps, None)
        im3 = ax3.imshow(ex3, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap=cmap, norm=norm, interpolation=interp)
        ax3.set_title("GP prediction - DUCB", fontsize=14)
        ax3.set_aspect("equal"); ax3.set_xlabel("East [m]"); ax3.set_ylabel("North [m]")

        # --- 4) RL ---
        ex4 = np.clip(Z_preds["RL"] - bg + eps, eps, None)
        im4 = ax4.imshow(ex4, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap=cmap, norm=norm, interpolation=interp)
        ax4.set_title("GP prediction - RL", fontsize=14)
        ax4.set_aspect("equal"); ax4.set_xlabel("East [m]"); ax4.set_ylabel("North [m]")

        fig.suptitle(f"{title_prefix}", fontsize=16)

        # place colorbar in reserved area
        cbar_ax = fig.add_axes([0.85, 0.08, 0.02, 0.80])  # [left, bottom, width, height]
        cbar = fig.colorbar(im1, cax=cbar_ax)

        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks[1:7])
        cbar.set_ticklabels(["≤550", "550.1"] + [f"{bg + t:.0f}" for t in ticks[3:7]], fontsize=14)
        cbar.set_label("Predicted concentration")
        
        plt.show()

    return results

def evaluate_strategies_for_field(
    base_model, likelihood,
    Xtest, ytrue,
    strategy_samples,
    results,
    obs_x=None, obs_y=None,
    make_plot=False,
    title_prefix="GP prediction"
):
    """
    Evaluate multiple sampling strategies on ONE true field, and optionally
    create a 2×2 plot of the truth and GP predictions.

    Parameters
    ----------
    base_model : ExactGP_RotMat32ARD
        Base GP model with fixed hyperparams (angle, ell_par, ell_perp, etc.).
    likelihood : gpytorch.likelihoods.GaussianLikelihood
    Xtest : torch.Tensor, shape (N_test, 2)
        Test coordinates (original coordinate frame).
    ytrue : torch.Tensor, shape (N_test,)
        True scalar field values at Xtest.
    strategy_samples : dict
        Mapping from strategy name -> (Xk, yk),
        where Xk, yk are torch tensors for that field.
        Example:
            {
              "Lawnmower": (X_lawn, y_lawn),
              "DUCB":      (X_ducb,  y_ducb),
              "RL":        (X_rl,    y_rl),
            }
    results : dict
        Results dict from init_strategy_results; updated in-place.
    obs_x, obs_y : int, optional
        Grid resolution of Xtest (so that N_test = obs_x * obs_y).
        Required if make_plot=True.
    make_plot : bool
        If True, also generate a 2×2 figure:
        [True field, Lawn­mower pred, DUCB pred, RL pred].
    title_prefix : str
        Prefix for the figure suptitle.

    Returns
    -------
    results : dict
        Updated results with appended RMSE and NLL for this field.
    """

    # Store predictions for plotting if needed
    preds_mu = {}

    for name, (Xk, yk) in strategy_samples.items():
        rmse, nll, mu, var = score_subset_with_swap(
            base_model, likelihood, Xk, yk, Xtest, ytrue
        )
        results[name]["rmse"].append(rmse)
        results[name]["nll"].append(nll)
        preds_mu[name] = mu.detach().cpu().numpy()

    # ----------------------------------------------------------------------
    # Optional plotting: 2×2 (truth + 3 strategy predictions)
    # ----------------------------------------------------------------------
    if make_plot:
        if obs_x is None or obs_y is None:
            raise ValueError("obs_x and obs_y must be provided when make_plot=True")

        # Convert truth and coords to numpy and reshape
        Xtest_np = Xtest.detach().cpu().numpy()
        ytrue_np = ytrue.detach().cpu().numpy()

        X = Xtest_np[:, 0].reshape(obs_y, obs_x)
        Y = Xtest_np[:, 1].reshape(obs_y, obs_x)
        Z_true = ytrue_np.reshape(obs_y, obs_x)

        # Get predictions in a fixed order (assuming keys exist)
        order = ["Lawnmower", "DUCB", "RL"]
        Z_preds = {name: preds_mu[name].reshape(obs_y, obs_x) for name in order}

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
        ax1, ax2, ax3, ax4 = axes.ravel()
        for ax in (ax1, ax2, ax3, ax4):
            ax.grid(False)  # disable major gridlines

        # 1) True field
        #im1 = ax1.pcolormesh(X, Y, Z_true, cmap="viridis", shading="auto")
        im1 = ax1.imshow(
            Z_true,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="viridis",
            interpolation="bicubic"  # <-- makes the image smooth
        )

        ax1.set_title("GP prediction - true field")
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label("Value")

        # 2) Lawnmower GP prediction
        Z_lawn = Z_preds["Lawnmower"]
        #im2 = ax2.pcolormesh(X, Y, Z_lawn, cmap="viridis", shading="auto")
        im2 = ax2.imshow(
            Z_lawn,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="viridis",
            interpolation="bicubic"  # <-- makes the image smooth
        )
        ax2.set_title("GP prediction – lawnmower")
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label("Predicted value")

        # 3) DUCB GP prediction
        Z_ducb = Z_preds["DUCB"]
        #im3 = ax3.pcolormesh(X, Y, Z_ducb, cmap="viridis", shading="auto")
        im3 = ax3.imshow(
            Z_ducb,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="viridis",
            interpolation="bicubic"  # <-- makes the image smooth
        )
        ax3.set_title("GP prediction – DUCB")
        ax3.set_aspect("equal", adjustable="box")
        ax3.set_xlabel("x"); ax3.set_ylabel("y")
        cbar3 = fig.colorbar(im3, ax=ax3)
        cbar3.set_label("Predicted value")

        # 4) RL GP prediction
        Z_rl = Z_preds["RL"]
        #im4 = ax4.pcolormesh(X, Y, Z_rl, cmap="viridis", shading="auto")
        im4 = ax4.imshow(
            Z_rl,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="viridis",
            interpolation="bicubic"  # <-- makes the image smooth
        )
        ax4.set_title("GP prediction – RL")
        ax4.set_aspect("equal", adjustable="box")
        ax4.set_xlabel("x"); ax4.set_ylabel("y")
        cbar4 = fig.colorbar(im4, ax=ax4)
        cbar4.set_label("Predicted value")

        fig.suptitle(f"{title_prefix}", fontsize=14)
        plt.tight_layout()
        plt.show()

    return results

def tighten_axis(ax, X, Y):
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)

def assemble_agent_plot_data(strategy_samples, agents="all"):
    """
    Assemble coords, measurements, and labels from a strategy_samples dict.

    Parameters
    ----------
    strategy_samples : dict
        Mapping: agent_name -> (coords, values)
        coords : (N,2) torch.Tensor or numpy array
        values : (N,)  torch.Tensor or numpy array
    agents : "all" or list of str
        If "all"  -> include every agent in strategy_samples.
        If list   -> only include those agent names (if present).

    Returns
    -------
    agent_coords_list : list of np.ndarray
    agent_vals_list   : list of np.ndarray
    agent_labels      : list of str
    """
    # --- Determine which agent labels to include ---
    if agents == "all":
        selected_agents = list(strategy_samples.keys())
    else:
        # filter only those that exist in the dict
        selected_agents = [a for a in agents if a in strategy_samples]

    agent_coords_list = []
    agent_vals_list   = []
    agent_labels      = []

    # --- Extract and convert to numpy ---
    for name in selected_agents:
        coords, vals = strategy_samples[name]

        # Convert coords
        if torch.is_tensor(coords):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.asarray(coords)

        # Convert vals
        if torch.is_tensor(vals):
            vals_np = vals.detach().cpu().numpy()
        else:
            vals_np = np.asarray(vals)

        agent_labels.append(name)
        agent_coords_list.append(coords_np)
        agent_vals_list.append(vals_np)

    return agent_coords_list, agent_vals_list, agent_labels

def plot_sampling_comparison_n_plots(
    env_xy, values,
    agent_coords_list,   # list of (N_i, 2) arrays
    agent_vals_list,     # list of (N_i,) arrays
    agent_labels,        # list of strings
    threshold,
    obs_x, obs_y,
    cmap="viridis",
    title="Sampling strategies vs true field",
    show_true_field=True,
):
    """
    Visual comparison of sampling paths for an arbitrary number of agents.

    Layout:
        If show_true_field=True:
            First panel = true scalar field
            Remaining = one panel per agent
        Otherwise:
            Only agent panels.

    All panels share:
        - LogNorm color scale
        - A single shared vertical colorbar

    Parameters
    ----------
    env_xy : array, shape (N, 2)
    values : array, shape (N,)
    """

    # reshape grid
    X = env_xy[:, 0].reshape(obs_y, obs_x).cpu().numpy()
    Y = env_xy[:, 1].reshape(obs_y, obs_x).cpu().numpy()
    Z_true = values.reshape(obs_y, obs_x).cpu().numpy()

    # ---------------- LogNorm Scaling ---------------- #
    bg  = threshold
    eps = 1e-2

    excess_true = np.clip(Z_true - bg + eps, eps, None)

    agent_excess_list = []
    for vals in agent_vals_list:
        vals_np = np.asarray(vals)
        if vals_np.size == 0:
            agent_excess_list.append(np.array([eps]))
        else:
            agent_excess_list.append(np.clip(vals_np - bg + eps, eps, None))

    vmax_data = np.nanmax([np.nanmax(excess_true)] +
                          [np.nanmax(ae) for ae in agent_excess_list])

    vmin = eps
    vmax = max(vmax_data, 2000 - bg + eps)

    norm = LogNorm(vmin=vmin, vmax=vmax)
    interp = "bicubic"

    # ---------------- Panel Count ---------------- #
    n_agents = len(agent_labels)
    total_panels = n_agents + (1 if show_true_field else 0)

    ncols = 2
    nrows = int(np.ceil(total_panels / ncols))

    fig_height = 3 + 3 * nrows
    fig = plt.figure(figsize=(12, fig_height), dpi=300)

    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        wspace=-0.38, hspace=0.40,
        left=0.07, right=0.92,
        bottom=0.08, top=0.95
    )

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            if len(axes) < total_panels:
                axes.append(fig.add_subplot(gs[r, c]))

    for ax in axes:
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=10)

    env_xy_flat = env_xy.reshape(-1, 2)

    # ---------------- Helper: scatter ---------------- #
    def scatter_samples(ax, X, Y, samp_xy, samp_vals, label):
        ax.scatter(X, Y, c="lightgray", s=5, alpha=0.3, linewidths=0)
        if samp_xy.size == 0:
            ax.set_title(label + " (no samples)", fontsize=13)
            return None

        excess = np.clip(samp_vals - bg + eps, eps, None)

        sc = ax.scatter(
            samp_xy[:, 0], samp_xy[:, 1],
            c=excess, cmap=cmap, norm=norm,
            s=12, linewidths=0
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("East [m]", fontsize=12)
        ax.set_ylabel("North [m]", fontsize=12)
        return sc

    # ---------------- Fill Panels ---------------- #
    panel_idx = 0

    # True field (optional)
    if show_true_field:
        ax = axes[panel_idx]
        panel_idx += 1

        # Flatten env_xy for background scatter (same grid as X, Y)
        env_xy_flat = env_xy.reshape(-1, 2)

        sc = scatter_samples(ax, X, Y, env_xy_flat, values, "True scalar field")
        ax.set_title("True scalar field", fontsize=13)
        ax.set_xlabel("East [m]", fontsize=12)
        ax.set_ylabel("North [m]", fontsize=12)
        mappable_for_cbar = sc
    else:
        mappable_for_cbar = None  # will be replaced with first agent scatter

    # Agent panels
    last_scatter = None
    for coords, vals, label in zip(agent_coords_list, agent_vals_list, agent_labels):
        if panel_idx >= len(axes):
            break
        ax = axes[panel_idx]
        panel_idx += 1

        sc = scatter_samples(ax, X, Y, np.asarray(coords), np.asarray(vals), label)
        if sc is not None:
            last_scatter = sc
            if not show_true_field and mappable_for_cbar is None:
                mappable_for_cbar = sc

    # ---------------- Shared Colorbar ---------------- #
    cbar_ax = fig.add_axes([0.85, 0.12, 0.02, 0.70])
    cbar = fig.colorbar(mappable_for_cbar, cax=cbar_ax)

    ticks = cbar.get_ticks()
    if len(ticks) > 7:
        ticks = ticks[1:7]
        cbar.set_ticks(ticks)

    ticklabels = [f"{bg + t:.0f}" for t in ticks]
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Concentration", fontsize=12)

    fig.suptitle(title, fontsize=14)
    plt.show()

def plot_sampling_comparison_lognorm_gridspec(
    env_xy, values,
    measurement_coords_lawnmower, measurements_lawnmower,
    ducb_coords, ducb_vals,
    rl_coords, rl_vals,
    obs_x, obs_y,            # grid resolution of env_xy (e.g., 100 × 100)
    cmap="viridis",
    title="Sampling strategies vs true field",
):
    """
    Creates a 2×2 figure (using GridSpec and LogNorm scaling):
      (1) True scalar field (smooth imshow with log-scaled excess above background)
      (2) Lawnmower samples (points colored by measurement, same log scale)
      (3) DUCB samples
      (4) RL samples

    Assumes env_xy is (N,2) covering a regular obs_y × obs_x grid.
    """

    # --- 1. Reshape true field into 2D grid --------------------------------
    X = env_xy[:, 0].reshape(obs_y, obs_x)
    Y = env_xy[:, 1].reshape(obs_y, obs_x)
    Z_true = values.reshape(obs_y, obs_x)

    lawn_xy  = np.asarray(measurement_coords_lawnmower)
    lawn_val = np.asarray(measurements_lawnmower)
    ducb_xy  = np.asarray(ducb_coords)
    ducb_val = np.asarray(ducb_vals)
    rl_xy    = np.asarray(rl_coords)
    rl_val   = np.asarray(rl_vals)

    # --- 2. LogNorm scaling parameters (same idea as GP plot) --------------
    bg  = 550.0    # background concentration
    eps = 1e-2     # small offset to avoid log(0)

    # Excess above background, clipped at eps
    excess_true  = np.clip(Z_true      - bg + eps, eps, None)
    excess_lawn  = np.clip(lawn_val    - bg + eps, eps, None)
    excess_ducb  = np.clip(ducb_val    - bg + eps, eps, None)
    excess_rl    = np.clip(rl_val      - bg + eps, eps, None)

    # Global vmax – either fixed or from data
    vmax_data = np.nanmax([
        np.nanmax(excess_true),
        np.nanmax(excess_lawn) if excess_lawn.size > 0 else eps,
        np.nanmax(excess_ducb) if excess_ducb.size > 0 else eps,
        np.nanmax(excess_rl)   if excess_rl.size > 0 else eps
    ])
    vmin = eps
    vmax = max(vmax_data, 2000 - bg + eps)  # follow your earlier choice

    norm = LogNorm(vmin=vmin, vmax=vmax)
    interp = "bicubic"

    # --- 3. Figure + GridSpec layout ---------------------------------------
    fig = plt.figure(figsize=(12, 9), dpi=300)
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        wspace=-0.28,   # tight horizontal spacing
        hspace=0.25,   # vertical spacing
        left=0.05,
        right=0.92,    # leave space on the right for colorbar
        bottom=0.07,
        top=0.90
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=10)

    # --- Helper: scatter sampling points with same norm --------------------
    def scatter_samples(ax, bg_xy, bg_z, samp_xy, samp_z, title):
        # light gray background grid points
        ax.scatter(bg_xy[:, 0], bg_xy[:, 1],
                   c="lightgray", s=4, alpha=0.3, linewidths=0)

        if samp_xy.size == 0:
            ax.set_title(title + " (no samples)", fontsize=13)
            ax.set_xlabel("East [m]", fontsize=12)
            ax.set_ylabel("North [m]", fontsize=12)
            return

        excess = np.clip(samp_z - bg + eps, eps, None)

        sc = ax.scatter(
            samp_xy[:, 0], samp_xy[:, 1],
            c=excess,
            cmap=cmap,
            norm=norm,
            s=12,
            linewidths=0,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("East [m]", fontsize=12)
        ax.set_ylabel("North [m]", fontsize=12)

        return sc

    # Flatten env_xy for background scatter (same grid as X, Y)
    env_xy_flat = env_xy.reshape(-1, 2)

    scatter_samples(ax1, env_xy_flat, Z_true.ravel(),
                    env_xy_flat, values,
                    "True scalar field")

    # --- 5. Panels (2–4): Sampling strategies ------------------------------
    scatter_samples(ax2, env_xy_flat, Z_true.ravel(),
                    lawn_xy, lawn_val,
                    "Lawnmower sampling")

    scatter_samples(ax3, env_xy_flat, Z_true.ravel(),
                    ducb_xy, ducb_val,
                    "DUCB sampling")

    sc4 = scatter_samples(ax4, env_xy_flat, Z_true.ravel(),
                          rl_xy, rl_val,
                          "RL sampling")

    # --- 6. Shared vertical colorbar on the right --------------------------
    # We'll anchor it with im1 (same norm & cmap as all panels)
    cbar_ax = fig.add_axes([0.85, 0.08, 0.02, 0.80])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc4, cax=cbar_ax)

    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks[1:7])
    # Relabel in terms of absolute concentration: bg + excess
    cbar.set_ticklabels(["≤550", "550.1"] + [f"{bg + t:.0f}" for t in ticks[3:7]], fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Concentration", fontsize=14)

    fig.suptitle(title, fontsize=16)
    plt.show()

def plot_sampling_comparison(
    env_xy, values,
    measurement_coords_lawnmower, measurements_lawnmower,
    ducb_coords, ducb_vals,
    rl_coords, rl_vals,
    obs_x, obs_y,            # grid resolution of env_xy (e.g., 100 × 100)
    cmap="viridis",
    title="Sampling strategies vs true field",
):
    """
    Creates a 2×2 figure:
      (1) True scalar field (smooth pcolormesh)
      (2) Lawnmower samples
      (3) DUCB samples
      (4) RL samples

    Assumes env_xy is (N,2) covering a regular obs_y × obs_x grid.
    """

    # reshape true field into 2D grid
    X = env_xy[:, 0].reshape(obs_y, obs_x)
    Y = env_xy[:, 1].reshape(obs_y, obs_x)
    Z = values.reshape(obs_y, obs_x)

    lawn_xy  = np.asarray(measurement_coords_lawnmower)
    lawn_val = np.asarray(measurements_lawnmower)
    ducb_xy  = np.asarray(ducb_coords)
    ducb_val = np.asarray(ducb_vals)
    rl_xy    = np.asarray(rl_coords)
    rl_val   = np.asarray(rl_vals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # ----------------------------------------------------------------------
    # 1) True environment (smooth 2D field)
    # ----------------------------------------------------------------------

    im1 = ax1.scatter(X, Y, c=Z, cmap=cmap, s=2)
    ax1.set_title("True scalar field")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Value (true)")

    # ----------------------------------------------------------------------
    # 2) Lawnmower sampling
    # ----------------------------------------------------------------------

    ax2.scatter(X, Y, c="lightgray", s=5, alpha=0.3, linewidths=0)
    im2 = ax2.scatter(lawn_xy[:, 0], lawn_xy[:, 1],
                      c=lawn_val, cmap=cmap,
                      vmin=np.nanmin(lawn_val), vmax=np.nanmax(lawn_val),
                      s=12, linewidths=0)
    ax2.set_title("Lawnmower sampling")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Measured value (Lawnmower)")

    # ----------------------------------------------------------------------
    # 3) DUCB sampling
    # ----------------------------------------------------------------------

    ax3.scatter(X, Y, c="lightgray", s=5, alpha=0.3, linewidths=0)
    im3 = ax3.scatter(ducb_xy[:, 0], ducb_xy[:, 1],
                      c=ducb_val, cmap=cmap,
                      vmin=np.nanmin(ducb_val), vmax=np.nanmax(ducb_val),
                      s=12, linewidths=0)
    ax3.set_title("DUCB sampling")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label("Measured value (DUCB)")

    # ----------------------------------------------------------------------
    # 4) RL sampling
    # ----------------------------------------------------------------------

    ax4.scatter(X, Y, c="lightgray", s=5, alpha=0.3, linewidths=0)
    im4 = ax4.scatter(rl_xy[:, 0], rl_xy[:, 1],
                      c=rl_val, cmap=cmap,
                      vmin=np.nanmin(rl_val), vmax=np.nanmax(rl_val),
                      s=12, linewidths=0)
    ax4.set_title("RL sampling")
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlabel("x"); ax4.set_ylabel("y")
    cbar4 = fig.colorbar(im4, ax=ax4)
    cbar4.set_label("Measured value (RL)")
    
    # layout
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_rmse_with_confidence_multi_ducb(rmse_lawn, rmse_rl_dict, rmse_ducb_dict, sample_points, mode='mean', save_str=''):
    """
    Plot RMSE mean ± 95% CI for Lawn mower and multiple DUCB agents.

    Parameters
    ----------
    rmse_lawn : torch.Tensor, shape (N_runs, K)
        RMSE across runs for the lawnmower strategy.
    rmse_ducb_dict : dict
        Mapping: ducb_name -> torch.Tensor of shape (N_runs, K)
    sample_points : array-like, shape (K,)
        Sample counts, e.g. [1000, 1200, 1400, 1600, 1800, 1948]
    """

    # Convert lawnmower
    L = rmse_lawn.cpu().numpy()
    #R = rmse_rl.cpu().numpy()

    # Convert RLs
    rl_np = {
        name: arr.cpu().numpy()
        for name, arr in rmse_rl_dict.items()
    }

    # Convert DUCBs
    ducb_np = {
        name: arr.cpu().numpy()
        for name, arr in rmse_ducb_dict.items()
    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    def add_curve(data, label, color, ls, alpha_fill=0.18, mode='mean'):
        if mode == 'mean':
            mean = data.mean(axis=0)
        else:
            mean = np.median(data, axis=0)

        std  = data.std(axis=0)
        print(f'{label}: {mean}')

        ci_low  = mean - 1.96 * std / np.sqrt(data.shape[0])
        ci_high = mean + 1.96 * std / np.sqrt(data.shape[0])

        ax.plot(sample_points, mean, label=label,
                color=color, linestyle=ls, lw=2)
        #ax.fill_between(sample_points, ci_low, ci_high, color=color, alpha=alpha_fill)

    # --- Lawn mower (reference) ---
    add_curve(L, "Lawnmower", "#1f77b4", "-", mode=mode)  # blue, solid
    #add_curve(R, "RL", "black", "--", mode=mode)
    # --- RL agents ---
    # Define a small set of grayscale-friendly styles to cycle through
    rl_colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    rl_lstyles = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]

    rl_names_sorted = sorted(rl_np.keys())  # stable order

    for i, name in enumerate(rl_names_sorted):
        data = rl_np[name]
        color = rl_colors[i % len(rl_colors)]
        ls    = rl_lstyles[i % len(rl_lstyles)]
        add_curve(data, name, color, ls, mode=mode)

    # --- DUCB agents ---
    # Define a small set of grayscale-friendly styles to cycle through
    ducb_colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    ducb_lstyles = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]

    ducb_names_sorted = sorted(ducb_np.keys())  # stable order

    for i, name in enumerate(ducb_names_sorted):
        data = ducb_np[name]
        color = ducb_colors[i % len(ducb_colors)]
        ls    = ducb_lstyles[i % len(ducb_lstyles)]
        add_curve(data, name, color, ls, mode=mode)

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE")
    ax.set_title("GP Prediction RMSE")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig('figures_p3/' + f'rmse_{mode}' + save_str + '.eps', format='eps', dpi=300)
    plt.show()

def plot_rmse_with_confidence(rmse_lawn, rmse_du, rmse_rl, sample_points):
    """
    rmse_*: tensors of shape (N_runs, K)  (e.g. K=6 intermediate sample sizes)
    sample_points: list of sample counts, e.g. [1000, 1200, 1400, 1600, 1800, 1948]
    """

    # convert to numpy
    L = rmse_lawn.cpu().numpy()
    D = rmse_du.cpu().numpy()
    R = rmse_rl.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # --- Mean and Confidence Intervals ---
    def add_curve(data, label, color, ls):
        mean = data.mean(axis=0)
        std  = data.std(axis=0)

        ci_low  = mean - 1.96 * std / np.sqrt(data.shape[0])
        ci_high = mean + 1.96 * std / np.sqrt(data.shape[0])

        ax.plot(sample_points, mean, label=label, color=color, linestyle=ls, lw=2)
        ax.fill_between(sample_points, ci_low, ci_high, color=color, alpha=0.25)

    add_curve(L, "Lawnmower", "#1f77b4", "-")    # blue, solid
    add_curve(D, "DUCB",      "#d62728", "--")   # red, dashed
    add_curve(R, "RL",        "#2ca02c", ":")    # green, dotted


    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE")
    ax.set_title("GP Prediction RMSE Across Strategies")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_cumsum_with_variance_multi_ducb(c_lawn, c_ducb_dict, c_rl_dict, ci=True, save_str=''):
    """
    Plot cumulative detections with mean ± std bands.

    Parameters
    ----------
    c_lawn : torch.Tensor, shape (N_runs, T)
        Cumulative detections for lawnmower.
    c_ducb_dict : dict[str, torch.Tensor]
        Mapping: DUCB name -> cumulative detections, shape (N_runs, T).
    c_rl : torch.Tensor, shape (N_runs, T)
        Cumulative detections for RL (padded to same T).
    """

    # --- Convert to numpy ---
    L = c_lawn.cpu().numpy()
    #R = c_rl.cpu().numpy()
    R_dict = {name: arr.cpu().numpy() for name, arr in c_rl_dict.items()}
    D_dict = {name: arr.cpu().numpy() for name, arr in c_ducb_dict.items()}

    T = L.shape[1]
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    def add_curve(data, label, color, ls, ci=True, alpha_fill=0.18):
        mean = data.mean(axis=0)

        ax.plot(x, mean, label=label, color=color, linestyle=ls, lw=2)
        if ci:
            var  = data.var(axis=0)
            std  = np.sqrt(var)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=alpha_fill
            )

    # --- Lawn mower (reference) ---
    add_curve(L, "Lawnmower", "#1f77b4", "-", ci)   # blue, solid
    # --- RL (as before) ---
    #add_curve(R, "RL", "black", "--")         # black, dashed
    # --- RL variants ---
    rl_colors  = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    rl_lstyles = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
    for i, name in enumerate(sorted(R_dict.keys())):
        data  = R_dict[name]
        color = rl_colors[i % len(rl_colors)]
        ls    = rl_lstyles[i % len(rl_lstyles)]
        add_curve(data, name, color, ls, ci)

    # --- DUCB variants ---
    ducb_colors  = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    ducb_lstyles = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for i, name in enumerate(sorted(D_dict.keys())):
        data  = D_dict[name]
        color = ducb_colors[i % len(ducb_colors)]
        ls    = ducb_lstyles[i % len(ducb_lstyles)]
        add_curve(data, name, color, ls, ci)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Cumulative detections (> threshold)")
    ax.set_title("Detection performance across strategies")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig('figures_p3/' + 'det_performances' + save_str + '.eps', format='eps', dpi=300)
    plt.show()

def plot_cumsum_with_variance(c_lawn, c_du, c_rl):
    """
    c_*: tensors of shape (N_runs, T)
    """

    L = c_lawn.cpu().numpy()
    D = c_du.cpu().numpy()
    R = c_rl.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    def add_curve(data, label, color, ls):
        mean = data.mean(axis=0)
        var  = data.var(axis=0)

        ax.plot(mean, label=label, color=color, linestyle=ls, lw=2)
        ax.fill_between(
            np.arange(len(mean)),
            mean - np.sqrt(var),
            mean + np.sqrt(var),
            color=color,
            alpha=0.25
        )

    add_curve(L, "Lawnmower", "#1f77b4", "-")    # blue, solid
    add_curve(D, "DUCB",      "#d62728", "--")   # red, dashed
    add_curve(R, "RL",        "#2ca02c", ":")    # green, dotted

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Cumulative Detections (> threshold)")
    ax.set_title("Detection Performance Across Strategies")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_rmse_and_cumsum_panels(rmse_lawn, rmse_du, rmse_rl,
                                c_lawn, c_du, c_rl,
                                sample_points):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ========== RMSE PANEL ==========
    ax = axes[0]

    def add_rmse_curve(ax, data, label, color, ls):
        d = data.cpu().numpy()
        mean = d.mean(axis=0)
        std  = d.std(axis=0)
        ci_low  = mean - 1.96 * std / np.sqrt(d.shape[0])
        ci_high = mean + 1.96 * std / np.sqrt(d.shape[0])

        ax.plot(sample_points, mean, label=label, linestyle=ls, lw=2, color=color)
        ax.fill_between(sample_points, ci_low, ci_high, color=color, alpha=0.18)

    add_rmse_curve(ax, rmse_lawn, "Lawnmower", "#1f77b4", "-")
    add_rmse_curve(ax, rmse_du,   "DUCB",      "#d62728", "--")
    add_rmse_curve(ax, rmse_rl,   "RL",        "#2ca02c", ":")



    ax.set_title("RMSE vs. Number of Samples")
    ax.set_xlabel("Samples")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # ========== CUMSUM PANEL ==========
    ax = axes[1]

    def add_cumsum_curve(ax, data, label, color, ls):
        d = data.cpu().numpy()
        mean = d.mean(axis=0)
        var  = d.var(axis=0)
        ax.plot(mean, label=label, linestyle=ls, lw=2, color=color)
        ax.fill_between(np.arange(len(mean)),
                        mean - np.sqrt(var),
                        mean + np.sqrt(var),
                        color=color, alpha=0.18)

    add_cumsum_curve(ax, c_lawn, "Lawnmower", "#1f77b4", "-")
    add_cumsum_curve(ax, c_du,   "DUCB",      "#d62728", "--")
    add_cumsum_curve(ax, c_rl,   "RL",        "#2ca02c", ":")

    ax.set_title("Cumulative Detections")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Cumulative (> threshold)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

def build_rmse_table_latex(
    rmse_lawnmower_all,
    rmse_ducb_all_stacked,   # dict: name -> tensor (N_runs, K)
    rmse_rl_all_stacked=None,        # optional: tensor (N_runs, K)
    sample_points=None,
    ci_level=1.96,           # 95% CI by default
    decimals=3,
    mode='mean'
):
    """
    Build a LaTeX table with RMSE mean ± CI for all agents.
    Lowest mean per sample point is bolded.

    Parameters
    ----------
    rmse_lawnmower_all : torch.Tensor, shape (N_runs, K)
    rmse_ducb_all_stacked : dict[str, torch.Tensor]
        Mapping agent_name -> (N_runs, K)
    rmse_rl_all : torch.Tensor or None
        If provided, included as an additional agent 'RL'.
    sample_points : list or array-like of length K
    ci_level : float
        Multiplier for std/sqrt(N) to get CI (1.96 for ~95%).
    decimals : int
        Number of decimal places in formatted output.

    Returns
    -------
    latex_str : str
        A LaTeX tabular environment as a string.
    """

    def stats_from_tensor(t: torch.Tensor, mode='mean'):
        t = t.float()
        if mode == 'mean':
            mean = t.mean(dim=0).cpu().numpy()
        else:
            mean = torch.quantile(t, q=0.5, dim=0).values.cpu().numpy()

        std  = t.std(dim=0, unbiased=True).cpu().numpy()
        n    = t.shape[0]
        ci   = ci_level * std / np.sqrt(n)
        return mean, ci

    # ---- Collect all agents and stats ----
    agent_stats = {}  # name -> (mean, ci)

    mean_lm, ci_lm = stats_from_tensor(rmse_lawnmower_all, mode=mode)
    agent_stats["Lawnmower"] = (mean_lm, ci_lm)

    #if rmse_rl_all is not None:
    #    mean_rl, ci_rl = stats_from_tensor(rmse_rl_all)
    #    agent_stats["RL"] = (mean_rl, ci_rl)
    for name, arr in rmse_rl_all_stacked.items():
        mean_rl, ci_rl = stats_from_tensor(arr, mode=mode)
        agent_stats[name] = (mean_rl, ci_rl)

    for name, arr in rmse_ducb_all_stacked.items():
        mean_du, ci_du = stats_from_tensor(arr, mode=mode)
        agent_stats[name] = (mean_du, ci_du)

    # ---- Determine best (lowest) mean per column ----
    agent_names = list(agent_stats.keys())
    K = agent_stats[agent_names[0]][0].shape[0]

    if sample_points is None:
        sample_points = list(range(K))

    # For each column j, find minimal mean across agents
    best_per_col = []
    for j in range(K):
        means_j = [agent_stats[name][0][j] for name in agent_names]
        min_val = min(means_j)
        best_per_col.append(min_val)

    # ---- Build LaTeX table ----
    # Header
    header_cols = "Agent"
    for sp in sample_points:
        header_cols += f" & {sp}"
    header_cols += r" \\"

    # Column alignment: one left, rest centered
    col_align = "l" + "c" * K

    lines = []
    lines.append(r"\begin{tabular}{" + col_align + "}")
    lines.append(r"\hline")
    lines.append(header_cols)
    lines.append(r"\hline")

    # Body
    for name in agent_names:
        mean, ci = agent_stats[name]

        # Escape underscores for LaTeX
        name_tex = name.replace("_", r"\_")

        row = [name_tex]
        for j in range(K):
            m = mean[j]
            c = ci[j]
            val_str = f"{m:.{decimals}f} $\pm$ {c:.{decimals}f}"

            # Bold if this is the best (lowest mean) for this column
            if np.isclose(m, best_per_col[j], rtol=1e-6, atol=1e-12):
                val_str = r"\textbf{" + val_str + "}"

            row.append(val_str)

        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    latex_str = "\n".join(lines)
    return latex_str


def build_rmse_grid_from_names(
    rmse_by_name: dict,
    kappas: np.ndarray,
    gammas: np.ndarray,
    *,
    name_prefix: str = "DUCB",
    kappa_scale_back: float = (255.0 / 5.0),
    kappa_fmt: str = ".2f",
    gamma_fmt: str = ".2f",
    mode = 'mean'
):
    """
    Build a 2D RMSE grid Z[gamma_idx, kappa_idx] from a name->RMSE dict.

    Returns
    -------
    Z : np.ndarray, shape (len(gammas_sorted), len(kappas_sorted))
        RMSE values (NaN if missing).
    kappas_sorted : np.ndarray
        Sorted kappas (x-axis).
    gammas_sorted : np.ndarray
        Sorted gammas (y-axis).
    """
    kappas = np.asarray(kappas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    kappas_sorted = np.sort(kappas)
    gammas_sorted = np.sort(gammas)

    Z = np.full((len(gammas_sorted), len(kappas_sorted)), np.nan, dtype=float)

    def make_name(k, g):
        k_str = format(k * kappa_scale_back, kappa_fmt)
        g_str = format(g, gamma_fmt)
        return f"{name_prefix}_k{k_str}_g{g_str}"

    for yi, g in enumerate(gammas_sorted):
        for xi, k in enumerate(kappas_sorted):
            name = make_name(k, g)
            if name in rmse_by_name:
                if mode == 'mean':
                    Z[yi, xi] = float(rmse_by_name[name][:, 5].mean())
                else:
                    Z[yi, xi] = float(torch.quantile(rmse_by_name[name][:, 5], q=0.5))
                    print(f'{name}: {Z[yi, xi]}')

    return Z, kappas_sorted, gammas_sorted


def plot_rmse_heatmap(
    Z: np.ndarray,
    kappas: np.ndarray,
    gammas: np.ndarray,
    *,
    title: str | None = None,
    xlabel: str = r"$\kappa$ (variance weight)",
    ylabel: str = r"$\gamma$ (distance weight)",
    cbar_label: str = "RMSE",
    cmap: str = "viridis_r",
    annotate: bool = True,
    annot_fmt: str = "{:.3g}",
    highlight_best: bool = False,
    nan_color: str = "#EEEEEE",
    figsize=(6.2, 4.8),
    dpi: int = 200,
    vmin=None,
    vmax=None,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot a publication-quality heatmap of RMSE values.
    """
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "figure.dpi": dpi,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=nan_color)

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(kappas)))
    ax.set_yticks(np.arange(len(gammas)))
    ax.set_xticklabels([f"{k:.3g}" for k in kappas])
    ax.set_yticklabels([f"{g:.3g}" for g in gammas])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Subtle grid between cells (journal-friendly)
    ax.set_xticks(np.arange(-0.5, len(kappas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(gammas), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if annotate:
        for yi in range(Z.shape[0]):
            for xi in range(Z.shape[1]):
                val = Z[yi, xi]
                if np.isfinite(val):
                    ax.text(
                        xi, yi,
                        annot_fmt.format(val),
                        ha="center", va="center",
                        fontsize=10,
                        color="black",
                    )

    if highlight_best and np.isfinite(Z).any():
        by, bx = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.scatter(
            [bx], [by],
            s=120, facecolors="none",
            edgecolors="black", linewidths=2
        )

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", format='eps')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def load_and_merge_results(
    filenames: List[str],
    *,
    device: str = "cpu",
    strict_metadata: bool = True,
):
    """
    Load multiple .pt result files and merge them into a single results dict.

    Merging rule:
    - tensors: concatenated along dim=0
    - dict[name -> tensor]: concatenated per name along dim=0
    - metadata: checked for consistency, then kept once

    Parameters
    ----------
    filenames : list of str
        Paths to .pt files saved via torch.save(results_dict).
    device : str
        Device to load tensors onto ("cpu" recommended).
    strict_metadata : bool
        If True, raises error if metadata differs across files.

    Returns
    -------
    merged : dict
        Merged results dictionary.
    """
    merged = {}
    metadata_keys = {
        "threshold",
        "n_samples_lim",
        "rl_agent_names",
        "ducb_agent_names",
    }

    # Containers for accumulating tensors
    tensor_accumulators = defaultdict(list)
    dict_tensor_accumulators = defaultdict(lambda: defaultdict(list))
    metadata_reference = {}

    for fname in filenames:
        data = torch.load(fname, map_location=device)

        for key, value in data.items():

            # --- metadata ---
            if key in metadata_keys:
                if key not in metadata_reference:
                    metadata_reference[key] = value
                elif strict_metadata and value != metadata_reference[key]:
                    raise ValueError(
                        f"Metadata mismatch for '{key}':\n"
                        f"{metadata_reference[key]} vs {value}"
                    )
                continue

            # --- plain tensors ---
            if torch.is_tensor(value):
                tensor_accumulators[key].append(value)

            # --- dict[name -> tensor] ---
            elif isinstance(value, dict):
                for subkey, tensor in value.items():
                    if not torch.is_tensor(tensor):
                        raise TypeError(
                            f"Expected tensor at {key}[{subkey}], got {type(tensor)}"
                        )
                    dict_tensor_accumulators[key][subkey].append(tensor)

            else:
                raise TypeError(
                    f"Unsupported type for key '{key}': {type(value)}"
                )

    # --- finalize tensor merges ---
    for key, tensor_list in tensor_accumulators.items():
        merged[key] = torch.cat(tensor_list, dim=0)

    # --- finalize dict[tensor] merges ---
    for key, subdict in dict_tensor_accumulators.items():
        merged[key] = {
            subkey: torch.cat(tensor_list, dim=0)
            for subkey, tensor_list in subdict.items()
        }

    # --- attach metadata ---
    merged.update(metadata_reference)

    return merged