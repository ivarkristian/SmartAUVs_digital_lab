# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.fft import rfftn, irfftn

# ---------------------------
# Kernel model semivariograms
# ---------------------------

def rbf_semivar_1d(h, sigma2, ell, nugget):
    # gamma(h) = sigma^2 (1 - exp(-h^2/(2 ell^2))) + nugget
    return sigma2 * (1.0 - np.exp(-0.5 * (h/ell)**2)) + nugget

def matern32_semivar_1d(h, sigma2, ell, nugget):
    # Matérn ν=3/2 correlation: ρ(h) = (1 + √3 h/ell) exp(-√3 h/ell)
    x = np.sqrt(3.0) * (h / ell)
    return sigma2 * (1.0 - (1.0 + x) * np.exp(-x)) + nugget

# ---------------------------
# Directional variogram core
# ---------------------------

def _autocovariance_fft(z):
    """Zero-mean autocovariance via FFT (circular)."""
    H, W = z.shape
    Z = rfftn(z)
    S = (Z * np.conj(Z)).real
    R = irfftn(S) / (H * W)
    return R

def _signed_displacements(H, W, dx, dy):
    """
    Build signed displacement grids (so angles cover [-pi, pi]).
    Uses circular (periodic) convention consistent with FFT autocovariance.
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # map to signed wrap-around distances
    yy = np.where(yy <= H // 2, yy, yy - H)
    xx = np.where(xx <= W // 2, xx, xx - W)
    dY = yy * dy
    dX = xx * dx
    return dX, dY

def _angular_difference_axial(angle, direction):
    """
    Smallest angular difference treating θ and θ+π as equivalent (axial).
    Returns value in [0, π/2].
    """
    # wrap difference to [-π, π]
    delta = np.arctan2(np.sin(angle - direction), np.cos(angle - direction))
    # axial symmetry: min(delta, π - delta) in [0, π/2]
    delta = np.abs(delta)
    return np.minimum(delta, np.pi - delta)

def directional_variograms(
    z, dx=1.0, dy=1.0, direction_deg=0.0, tol_deg=22.5, nbins=40, maxlag=None
):
    """
    Compute empirical directional semivariograms parallel and perpendicular
    to a given direction (degrees from +x axis, CCW).
    Returns (h_par, gamma_par), (h_perp, gamma_perp).
    """
    z = np.asarray(z, dtype=float)
    z = z - np.nanmean(z)
    H, W = z.shape

    R = _autocovariance_fft(z)          # covariance grid
    C0 = R[0, 0]

    dX, dY = _signed_displacements(H, W, dx, dy)
    dist = np.hypot(dX, dY)
    angle = np.arctan2(dY, dX)          # [-π, π]

    if maxlag is None:
        maxlag = 0.5 * min(H * dy, W * dx)

    direction = np.deg2rad(direction_deg)
    tol = np.deg2rad(tol_deg)

    # axial angular difference wrt direction
    delta = _angular_difference_axial(angle, direction)

    # masks for parallel and perpendicular directions
    mask_par = (delta <= tol)
    mask_perp = (np.abs(delta - np.pi/2) <= tol)

    # exclude zero-lag from semivariogram estimation
    mask_nonzero = (dist > 0)
    mask_par &= mask_nonzero
    mask_perp &= mask_nonzero

    # radial binning
    h_edges = np.linspace(0, maxlag, nbins + 1)
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])

    def bin_semivar(mask_dir):
        emp = np.full(nbins, np.nan)
        for i in range(nbins):
            m = mask_dir & (dist >= h_edges[i]) & (dist < h_edges[i+1])
            if np.any(m):
                # gamma(h) = C(0) - mean C(h)
                emp[i] = C0 - np.nanmean(R[m])
        good = np.isfinite(emp)
        return h_centers[good], emp[good]

    h_par, gamma_par = bin_semivar(mask_par)
    h_perp, gamma_perp = bin_semivar(mask_perp)

    return (h_par, gamma_par), (h_perp, gamma_perp)

# %%
import numpy as np
from scipy.optimize import curve_fit

def detrend_mean(z):
    return z - np.nanmean(z)

def radial_variogram(z, dx=1.0, dy=1.0, nbins=40, use_semivariogram=True, maxlag=None):
    """z: 2D array; returns bin centers (h) and empirical gamma(h) or C(h)."""
    z = np.asarray(z)
    z = detrend_mean(z)
    H, W = z.shape
    # Build all lags via FFT-based correlation for speed
    Z = np.fft.rfftn(z)
    S = (Z * np.conj(Z)).real       # power spectrum
    R = np.fft.irfftn(S) / (H*W)    # autocovariance on grid (zero-mean)
    # Build distance grid relative to origin
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    yy = np.where(yy<=H//2, yy, H-yy)
    xx = np.where(xx<=W//2, xx, W-xx)
    d = np.sqrt((yy*dy)**2 + (xx*dx)**2)

    if maxlag is None:
        maxlag = 0.5*min(H*dy, W*dx)

    # Radial binning
    h_edges = np.linspace(0, maxlag, nbins+1)
    h_centers = 0.5*(h_edges[:-1]+h_edges[1:])
    emp = np.zeros(nbins)
    counts = np.zeros(nbins, int)
    for i in range(nbins):
        m = (d>=h_edges[i]) & (d<h_edges[i+1])
        if use_semivariogram:
            # gamma(h) = C(0) - C(h)
            C0 = R[0,0]
            emp[i] = C0 - np.nanmean(R[m]) if m.any() else np.nan
        else:
            emp[i] = np.nanmean(R[m]) if m.any() else np.nan
        counts[i] = m.sum()

    # Clean NaNs
    good = np.isfinite(emp) & (counts>0)
    return h_centers[good], emp[good]

# Fit RBF semivariogram: gamma(h) = sigma2 * (1 - exp(-h^2/(2*l^2))) + nugget
def rbf_semivar(h, sigma2, ell, nugget):
    return sigma2 * (1.0 - np.exp(-h**2/(2*ell**2))) + nugget


# ---------------------------
# Fit + Plot wrappers
# ---------------------------

# %%
import matplotlib.pyplot as plt
def fit_and_plot_variogram(z, dx=1.0, dy=1.0, nbins=50):
    """Compute, fit, and plot the variogram with RBF kernel."""
    h, gamma_emp = radial_variogram(z, dx, dy, nbins)

    # Initial guesses and fit
    p0 = [np.nanmax(gamma_emp), np.median(h), 0.0]
    popt, _ = curve_fit(rbf_semivar, h, gamma_emp, p0=p0, bounds=(0, np.inf))
    sigma2, ell, nugget = popt
    print(f"Estimated RBF lengthscale ℓ ≈ {ell:.3f}")

    # --- Plot ---
    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(h, gamma_emp, s=25, c="tab:blue", label="Empirical", alpha=0.7)
    h_fit = np.linspace(0, h.max(), 200)
    plt.plot(h_fit, rbf_semivar(h_fit, *popt), c="tab:red", lw=2.5,
             label=f"Fitted RBF (ℓ={ell:.2f})")
    plt.xlabel("Lag distance")
    plt.ylabel("Semivariance")
    plt.title("Empirical and Fitted Variogram")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ell, sigma2, nugget

def matern_semivar_1d(h, sigma2, ell, nugget, nu):
    """
    1D Matérn semivariogram model:
        γ(h) = nugget + sigma2 * [1 - ρ(h)]

    where ρ(h) is the Matérn correlation with smoothness nu ∈ {0.5, 1.5, 2.5}.

    Parameters
    ----------
    h : array-like
        Lag distances.
    sigma2 : float
        Partial sill (structured variance).
    ell : float
        Correlation lengthscale.
    nugget : float
        Nugget.
    nu : float
        Smoothness parameter (0.5, 1.5, or 2.5 supported).
    """
    h = np.asarray(h, dtype=float)
    r = h / ell

    if nu == 0.5:
        # Matérn 1/2: ρ(r) = exp(-r)
        corr = np.exp(-r)
    elif nu == 1.5:
        # Matérn 3/2: ρ(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)
        c = np.sqrt(3.0)
        cr = c * r
        corr = (1.0 + cr) * np.exp(-cr)
    elif nu == 2.5:
        # Matérn 5/2: ρ(r) = (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)
        c = np.sqrt(5.0)
        cr = c * r
        corr = (1.0 + cr + (cr ** 2) / 3.0) * np.exp(-cr)
    else:
        raise ValueError(f"Unsupported nu={nu}. Use 0.5, 1.5, or 2.5.")

    return nugget + sigma2 * (1.0 - corr)

def fit_and_plot_anisotropic_variogram(
    z, dx=1.0, dy=1.0,
    direction_deg=0.0, tol_deg=22.5,
    nbins=40,
    kernel="matern32",      # "rbf", "matern12", "matern32", "matern52", or "matern"
    nu=1.5,                 # used if kernel starts with "matern" and no explicit ν is encoded
    plot=True, report=True
):
    """
    Compute directional empirical variograms parallel & perpendicular to 'direction_deg',
    fit 1D kernel model to each, and plot both.

    Returns:
        (ell_par, ell_perp, sigma2_par, sigma2_perp, nugget_par, nugget_perp)

    Parameters
    ----------
    z : 2D array
        Scalar field.
    dx, dy : float
        Grid spacing in physical units.
    direction_deg : float
        Main direction (e.g. current direction), degrees CCW from +x.
    tol_deg : float
        Angular tolerance around direction_deg / perpendicular for binning.
    nbins : int
        Number of lag bins.
    kernel : str
        "rbf" or one of:
          - "matern12"  (ν=0.5)
          - "matern32"  (ν=1.5, default)
          - "matern52"  (ν=2.5)
          - "matern"    (use the nu argument)
    nu : float
        Smoothness parameter if kernel starts with "matern" and does not encode ν explicitly.
        Allowed: 0.5, 1.5, 2.5.
    """

    # 1) empirical directional variograms
    (h_par, g_par), (h_perp, g_perp) = directional_variograms(
        z, dx, dy, direction_deg, tol_deg, nbins
    )

    kernel_l = kernel.lower()

    if kernel_l == "rbf":
        # Use your existing RBF semivariogram model
        model = rbf_semivar_1d
        label_kernel = "RBF"
        # sensible initials: [sigma2, ell, nugget]
        p0_par  = [np.nanmax(g_par)  if g_par.size  else 1.0,
                   np.median(h_par)  if h_par.size  else 1.0,
                   0.0]
        p0_perp = [np.nanmax(g_perp) if g_perp.size else 1.0,
                   np.median(h_perp) if h_perp.size else 1.0,
                   0.0]

    else:
        # Matérn family ----------------------------------------
        # Decide ν from kernel string if encoded
        if "12" in kernel_l:
            nu_local = 0.5
        elif "32" in kernel_l:
            nu_local = 1.5
        elif "52" in kernel_l:
            nu_local = 2.5
        else:
            nu_local = float(nu)

        if nu_local not in (0.5, 1.5, 2.5):
            raise ValueError(f"Unsupported nu={nu_local}. Use 0.5, 1.5, or 2.5.")

        # build a Matérn model closure with this ν
        def matern_model(h, sigma2, ell, nugget):
            return matern_semivar_1d(h, sigma2, ell, nugget, nu=nu_local)

        model = matern_model
        label_kernel = f"Matérn ν={nu_local:g}"
        p0_par  = [np.nanmax(g_par)  if g_par.size  else 1.0,
                   np.median(h_par)  if h_par.size  else 1.0,
                   0.0]
        p0_perp = [np.nanmax(g_perp) if g_perp.size else 1.0,
                   np.median(h_perp) if h_perp.size else 1.0,
                   0.0]

    # 2) fit each direction
    def safe_fit(h, g, p0):
        if h.size < 3:
            return (np.nan, np.nan, np.nan), None
        popt, pcov = curve_fit(model, h, g, p0=p0, bounds=(0, np.inf))
        return popt, pcov

    (sigma2_par,  ell_par,  nugget_par),  _ = safe_fit(h_par,  g_par,  p0_par)
    (sigma2_perp, ell_perp, nugget_perp), _ = safe_fit(h_perp, g_perp, p0_perp)

    # 3) plots
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
        for ax, (h, g, sigma2, ell, nugget, title) in zip(
            axes,
            [
                (h_par,  g_par,  sigma2_par,  ell_par,  nugget_par,
                 f"Parallel (dir={direction_deg:.1f}°)"),
                (h_perp, g_perp, sigma2_perp, ell_perp, nugget_perp,
                 "Perpendicular"),
            ],
        ):
            ax.scatter(h, g, s=25, color="tab:blue", alpha=0.75, label="Empirical")
            if np.isfinite(ell):
                h_fit = np.linspace(0, h.max() if h.size else 1.0, 200)
                ax.plot(h_fit, model(h_fit, sigma2, ell, nugget),
                        color="tab:red", lw=2.2,
                        label=f"Fitted {label_kernel}\nℓ={ell:.2f}")
            ax.set_xlabel("Lag distance")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=9)

        axes[0].set_ylabel("Semivariance")
        fig.suptitle(f"Directional Variograms (tol ±{tol_deg:.1f}°)", y=1.03, fontsize=13)
        plt.tight_layout()
        plt.show()

    # 4) report anisotropy
    if report:
        if np.isfinite(ell_par) and np.isfinite(ell_perp):
            ratio = ell_par / ell_perp if ell_perp > 0 else np.nan
            print(f"{label_kernel} directional lengthscales:")
            print(f"  ℓ_parallel      ≈ {ell_par:.3f},  σ²_par = {sigma2_par:.3f}")
            print(f"  ℓ_perpendicular ≈ {ell_perp:.3f}, σ²_perp = {sigma2_perp:.3f}")
            print(f"  Anisotropy ratio (ℓ∥ / ℓ⊥) ≈ {ratio:.3f}")
        else:
            print("Could not fit lengthscales (insufficient or degenerate data in a direction).")

    return ell_par, ell_perp, sigma2_par, sigma2_perp, nugget_par, nugget_perp