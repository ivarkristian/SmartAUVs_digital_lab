import math
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
    """
    Set (Xk, yk) as the training data of `model` and evaluate RMSE + NLL on Xtest.
    This is simpler and more robust than get_fantasy_model for your comparison.
    """
    model.set_train_data(inputs=Xk, targets=yk, strict=False)
    model.eval(); likelihood.eval()

    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(Xtest))

    mu, var = pred.mean, pred.variance
    rmse = torch.sqrt(torch.mean((mu - ytrue)**2)).item()
    nll = -pred.log_prob(ytrue).mean().item()
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
def evaluate_strategies_for_field_lognorm(
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
        
        bg = 550.0   # your background concentration (or median/mean)
        eps = 1e-3   # small offset to avoid log(0)

        # Shift field so that bg → ≈0, then clip
        excess = np.clip(Z_true - bg + eps, eps, None)

        vmin = eps
        vmax = np.max(excess)

        # 1) True field
        #im1 = ax1.pcolormesh(X, Y, Z_true, cmap="viridis", shading="auto")
        im1 = ax1.imshow(
            excess,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="bicubic"  # <-- makes the image smooth
        )

        ax1.set_title("True field")
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
        ax2.set_title("GP prediction – Lawnmower")
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

        ax1.set_title("True field")
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
        ax2.set_title("GP prediction – Lawnmower")
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
    
    tighten_axis(ax1, X, Y)
    tighten_axis(ax2, X, Y)
    tighten_axis(ax3, X, Y)
    tighten_axis(ax4, X, Y)
    
    # layout
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
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