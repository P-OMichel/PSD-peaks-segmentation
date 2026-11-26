import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF, FOOOFGroup

# ==============================================================================
# 0. PUBLICATION STYLING (TIMES NEW ROMAN / SERIF & COMPACT FONT SIZES)
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Times"],
    "mathtext.fontset": "stix",  # Matches Times New Roman closely for LaTeX equations
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "figure.dpi": 150,
})

# ==============================================================================
# 1. GENERATIVE MODELS: LOG-ADDITIVE GAUSSIAN VS. PSEUDO-VOIGT
# ==============================================================================
def periodic_bump(freqs, center, height, sigma, shape="gaussian", eta=0.5):
    """
    Evaluates a periodic bump parameterized by Gaussian standard deviation `sigma`
    so both shapes share the exact same Full Width at Half Maximum (FWHM):
        FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma

    - 'gaussian':     height * exp(-(f - center)^2 / (2 * sigma^2))
    - 'pseudo_voigt': height * [ (1 - eta) * Gaussian(f; sigma) + eta * Lorentzian(f; gamma = FWHM / 2) ]
    """
    g = np.exp(-((freqs - center) ** 2) / (2.0 * sigma**2))
    if shape == "gaussian":
        return height * g
    elif shape == "pseudo_voigt":
        gamma = sigma * np.sqrt(2.0 * np.log(2.0))  # HWHM matching Gaussian FWHM
        l = 1.0 / (1.0 + ((freqs - center) / gamma) ** 2)
        return height * ((1.0 - eta) * g + eta * l)
    else:
        raise ValueError("shape must be 'gaussian' or 'pseudo_voigt'")

def generate_log_psd(freqs, offset, exponent, peaks, shape="gaussian", eta=0.5):
    """
    Generates a noiseless PSD with bumps added directly in log10 power space:
        log10(P) = offset - exponent * log10(f) + sum(Bump)
    """
    aperiodic_log = offset - exponent * np.log10(freqs)

    periodic_log = np.zeros_like(freqs)
    for cf, height, sigma in peaks:
        periodic_log += periodic_bump(freqs, cf, height, sigma, shape=shape, eta=eta)

    psd_log = aperiodic_log + periodic_log
    psd_lin = 10**psd_log
    return psd_lin, psd_log, aperiodic_log

def fit_single_exponent(freqs, psd_linear, fooof_kwargs, freq_range):
    """Fits a single PSD and returns the FOOOF object and estimated exponent."""
    fm = FOOOF(**fooof_kwargs)
    fm.fit(freqs, psd_linear, freq_range=freq_range)
    return fm, fm.get_params("aperiodic_params", "exponent")

def fit_batch_exponents(freqs, spectra_linear, fooof_kwargs, freq_range):
    """Fits a 2D array of linear PSDs with FOOOFGroup and returns estimated exponents."""
    fg = FOOOFGroup(**fooof_kwargs)
    fg.fit(freqs, np.asarray(spectra_linear), freq_range=freq_range)
    return fg.get_params("aperiodic_params", "exponent")

# ==============================================================================
# 2. GLOBAL PARAMETERS & FOOOF SETTINGS (EXAGGERATED AMPLITUDE & WIDTH)
# ==============================================================================
freqs = np.arange(0.5, 45.1, 0.2)
freq_range = [0.5, 45.0]

true_offset = 2.0
true_exponent = 3.8
voigt_eta = 0.5  # 50% Gaussian + 50% Lorentzian mixing weight

# Fixed low-frequency (delta) peak: cf=2.5 Hz, a_1=2.5 log10, sigma_1=0.8 Hz (FWHM=1.88 Hz)
# Remains well inside [0.5, 45.0] Hz without edge-cutting at 0.5 Hz
delta_peak = (2.5, 2.5, 0.8)

# Exaggerated second peak at 14.0 Hz:
# sigma_2 = 2.2 Hz -> FWHM = 5.18 Hz (comfortably inside peak_width_limits=[0.5, 16.0]
# and >1.8 FWHM away from both 0.5 Hz and 45.0 Hz boundaries)
alpha_cf = 14.0
ex_alpha_sigma = 2.2
ex_alpha_height = 4.0  # +4.0 log10 power above 1/f

fooof_kwargs = dict(
    peak_width_limits=[0.5, 16.0],  # Accommodates FWHM up to ~6.6 Hz (sigma=2.8 Hz) cleanly
    max_n_peaks=6,                  # Ample slots so Pseudo-Voigt skirt-splitting never clips
    min_peak_height=0.1,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
    verbose=False,
)

# ==============================================================================
# 3. SINGLE EXAMPLES: 1 BUMP VS. 2 BUMPS (GAUSSIAN & PSEUDO-VOIGT)
# ==============================================================================
peaks_1 = [delta_peak]
peaks_2 = [delta_peak, (alpha_cf, ex_alpha_height, ex_alpha_sigma)]

# --- Gaussian ---
psd_g1_lin, psd_g1_log, true_ap_log = generate_log_psd(
    freqs, true_offset, true_exponent, peaks_1, shape="gaussian"
)
fm_g1, exp_g1 = fit_single_exponent(freqs, psd_g1_lin, fooof_kwargs, freq_range)

psd_g2_lin, psd_g2_log, _ = generate_log_psd(
    freqs, true_offset, true_exponent, peaks_2, shape="gaussian"
)
fm_g2, exp_g2 = fit_single_exponent(freqs, psd_g2_lin, fooof_kwargs, freq_range)

# --- Pseudo-Voigt ---
psd_pv1_lin, psd_pv1_log, _ = generate_log_psd(
    freqs, true_offset, true_exponent, peaks_1, shape="pseudo_voigt", eta=voigt_eta
)
fm_pv1, exp_pv1 = fit_single_exponent(freqs, psd_pv1_lin, fooof_kwargs, freq_range)

psd_pv2_lin, psd_pv2_log, _ = generate_log_psd(
    freqs, true_offset, true_exponent, peaks_2, shape="pseudo_voigt", eta=voigt_eta
)
fm_pv2, exp_pv2 = fit_single_exponent(freqs, psd_pv2_lin, fooof_kwargs, freq_range)

# ==============================================================================
# 4. PARAMETER SWEEPS: EXAGGERATED 2ND PEAK WIDTH (SIGMA) & AMPLITUDE
# ==============================================================================
n_grid = 35
# sigma_2 up to 2.8 Hz -> FWHM = 6.59 Hz (safely well below peak_width_limits[1] = 16.0 Hz)
sigma_grid = np.linspace(0.5, 2.8, n_grid)
# a_2 up to 5.5 log10 power units
amp_grid = np.linspace(0.5, 5.5, n_grid)

# --- Sweep 1: Second Peak Width (Matched FWHM via Sigma) ---
def sweep_sigma(shape):
    psds = [
        generate_log_psd(
            freqs, true_offset, true_exponent, [delta_peak, (alpha_cf, ex_alpha_height, s)],
            shape=shape, eta=voigt_eta
        )[0]
        for s in sigma_grid
    ]
    return fit_batch_exponents(freqs, psds, fooof_kwargs, freq_range) - true_exponent

err_sig_g = sweep_sigma("gaussian")
err_sig_pv = sweep_sigma("pseudo_voigt")

# --- Sweep 2: Second Peak Amplitude ---
def sweep_amp(shape):
    psds = [
        generate_log_psd(
            freqs, true_offset, true_exponent, [delta_peak, (alpha_cf, h, ex_alpha_sigma)],
            shape=shape, eta=voigt_eta
        )[0]
        for h in amp_grid
    ]
    return fit_batch_exponents(freqs, psds, fooof_kwargs, freq_range) - true_exponent

err_amp_g = sweep_amp("gaussian")
err_amp_pv = sweep_amp("pseudo_voigt")

# ==============================================================================
# 5. VISUALIZATION (4 ROWS: BUMP PROFILE + 1-BUMP + 2-BUMP + SWEEPS + LATEX)
# ==============================================================================
fig = plt.figure(figsize=(15.5, 17.5), dpi=150)

gs = fig.add_gridspec(
    4, 2,
    hspace=0.52,
    wspace=0.25,
    top=0.95,
    bottom=0.14,
    left=0.07,
    right=0.96,
)

# --- ROW 1: ISOLATED BUMP SHAPE COMPARISON (LINEAR Y vs. LOG10 Y) ---
f_zoom = np.linspace(2.0, 26.0, 500)
bump_g = periodic_bump(f_zoom, alpha_cf, ex_alpha_height, ex_alpha_sigma, shape="gaussian")
bump_pv = periodic_bump(f_zoom, alpha_cf, ex_alpha_height, ex_alpha_sigma, shape="pseudo_voigt", eta=voigt_eta)
fwhm_val = 2.0 * np.sqrt(2.0 * np.log(2.0)) * ex_alpha_sigma

# Panel (A): Bump Profile (Linear Scale of log10-Power Addition)
ax_b1 = fig.add_subplot(gs[0, 0])
ax_b1.plot(f_zoom, bump_g, color="seagreen", lw=1.8, label=f"Gaussian ($\\sigma_2={ex_alpha_sigma}$ Hz, $a_2={ex_alpha_height}$)")
ax_b1.plot(f_zoom, bump_pv, color="teal", ls="--", lw=1.8, label=f"Pseudo-Voigt ($\\eta={voigt_eta}$, Matched FWHM)")
ax_b1.axhline(ex_alpha_height / 2.0, color="gray", ls=":", lw=1.0, label=f"Half-Max (FWHM = {fwhm_val:.2f} Hz)")
ax_b1.axvline(alpha_cf - fwhm_val / 2.0, color="gray", ls=":", lw=0.9)
ax_b1.axvline(alpha_cf + fwhm_val / 2.0, color="gray", ls=":", lw=0.9)
ax_b1.set_title("(A) Isolated Bump Profile: Linear $\\Delta\\log_{10}P$ Scale", fontweight="bold", pad=8)
ax_b1.set_xlabel("Frequency (Hz)", labelpad=4)
ax_b1.set_ylabel("Added Height $\\Delta\\log_{10}P$", labelpad=4)
ax_b1.set_xlim(2.0, 26.0)
ax_b1.grid(True, ls=":", alpha=0.55)
ax_b1.legend(loc="upper right", frameon=True, framealpha=0.9)

# Panel (B): Bump Profile (Log10 Scale of Added Height — Exposes Heavy Tails)
ax_b2 = fig.add_subplot(gs[0, 1])
ax_b2.plot(f_zoom, bump_g, color="seagreen", lw=1.8, label="Gaussian (Exponential Tail Drop-Off)")
ax_b2.plot(f_zoom, bump_pv, color="teal", ls="--", lw=1.8, label="Pseudo-Voigt (Heavy Algebraic Tails)")
ax_b2.fill_between(f_zoom, bump_g, bump_pv, where=(bump_pv > bump_g), color="teal", alpha=0.14, label="Unmodeled Heavy-Tail Excess")
ax_b2.set_yscale("log")
ax_b2.set_ylim(1e-4, 8.0)
ax_b2.set_xlim(2.0, 26.0)
ax_b2.set_title("(B) Isolated Bump Profile: Log Scale (Tail Comparison)", fontweight="bold", pad=8)
ax_b2.set_xlabel("Frequency (Hz)", labelpad=4)
ax_b2.set_ylabel("Added Height $\\Delta\\log_{10}P$ (log scale)", labelpad=4)
ax_b2.grid(True, which="both", ls=":", alpha=0.5)
ax_b2.legend(loc="upper right", frameon=True, framealpha=0.9)

# Helper for PSD example plots
def plot_example(ax, psd_log, fm, exp_est, title, est_color):
    ax.plot(freqs, psd_log, color="black", lw=1.4, label="Noiseless PSD")
    ax.plot(fm.freqs, fm.fooofed_spectrum_, color="crimson", ls="--", lw=1.4, label="FOOOF Fit")
    ax.plot(freqs, true_ap_log, "b--", lw=1.4, label=f"True $1/f$ ($\\chi={true_exponent:.2f}$)")
    ax.plot(fm.freqs, fm._ap_fit, color=est_color, ls=":", lw=2.2, label=f"Est $1/f$ ($\\hat{{\\chi}}={exp_est:.3f}$)")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Frequency (Hz)", labelpad=4)
    ax.set_ylabel("$\\log_{10}$ Power", labelpad=4)
    ax.set_xlim(0.5, 45)
    ax.grid(True, ls=":", alpha=0.55)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

# --- ROW 2: 1 BUMP (DELTA ONLY) — GAUSSIAN VS. PSEUDO-VOIGT ---
plot_example(
    fig.add_subplot(gs[1, 0]),
    psd_g1_log, fm_g1, exp_g1,
    f"(C) 1 Bump (Delta Only, $a_1={delta_peak[1]}$): Gaussian  |  Error = {exp_g1 - true_exponent:+.3f}",
    "seagreen",
)
plot_example(
    fig.add_subplot(gs[1, 1]),
    psd_pv1_log, fm_pv1, exp_pv1,
    f"(D) 1 Bump (Delta Only, $a_1={delta_peak[1]}$): Pseudo-Voigt ($\\eta={voigt_eta}$)  |  Error = {exp_pv1 - true_exponent:+.3f}",
    "teal",
)

# --- ROW 3: 2 BUMPS (DELTA + 14 HZ) — GAUSSIAN VS. PSEUDO-VOIGT ---
plot_example(
    fig.add_subplot(gs[2, 0]),
    psd_g2_log, fm_g2, exp_g2,
    f"(E) 2 Bumps ($a_2={ex_alpha_height}, \\sigma_2={ex_alpha_sigma}$): Gaussian  |  Error = {exp_g2 - true_exponent:+.3f}",
    "seagreen",
)
plot_example(
    fig.add_subplot(gs[2, 1]),
    psd_pv2_log, fm_pv2, exp_pv2,
    f"(F) 2 Bumps ($a_2={ex_alpha_height}, \\sigma_2={ex_alpha_sigma}$): Pseudo-Voigt ($\\eta={voigt_eta}$)  |  Error = {exp_pv2 - true_exponent:+.3f}",
    "teal",
)

# --- ROW 4: ERROR SWEEPS (SIGMA & AMPLITUDE) ---
ax_sig = fig.add_subplot(gs[3, 0])
ax_sig.axhline(0, color="black", ls="--", lw=1.0, alpha=0.7)
ax_sig.plot(sigma_grid, err_sig_g, "o-", color="seagreen", markersize=3.5, lw=1.3, label="Gaussian Bumps")
ax_sig.plot(sigma_grid, err_sig_pv, "s--", color="teal", markersize=3.5, lw=1.3, label=f"Pseudo-Voigt Bumps ($\\eta={voigt_eta}$)")
ax_sig.set_title(f"(G) Exponent Error vs. 2nd Peak Width ($\\sigma_2$, Fixed $a_2={ex_alpha_height}$)", fontweight="bold", pad=8)
ax_sig.set_xlabel("2nd Peak Equivalent Standard Deviation $\\sigma_2$ (Hz)", labelpad=4)
ax_sig.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{\\mathrm{true}}$)", labelpad=4)
ax_sig.grid(True, ls=":", alpha=0.55)
ax_sig.legend(loc="lower left", frameon=True, framealpha=0.9)

ax_amp = fig.add_subplot(gs[3, 1])
ax_amp.axhline(0, color="black", ls="--", lw=1.0, alpha=0.7)
ax_amp.plot(amp_grid, err_amp_g, "o-", color="seagreen", markersize=3.5, lw=1.3, label="Gaussian Bumps")
ax_amp.plot(amp_grid, err_amp_pv, "s--", color="teal", markersize=3.5, lw=1.3, label=f"Pseudo-Voigt Bumps ($\\eta={voigt_eta}$)")
ax_amp.set_title(f"(H) Exponent Error vs. 2nd Peak Amplitude ($a_2$, Fixed $\\sigma_2={ex_alpha_sigma}$ Hz)", fontweight="bold", pad=8)
ax_amp.set_xlabel("2nd Peak Amplitude $a_2$ ($\\log_{10}$ Power)", labelpad=4)
ax_amp.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{\\mathrm{true}}$)", labelpad=4)
ax_amp.grid(True, ls=":", alpha=0.55)
ax_amp.legend(loc="lower left", frameon=True, framealpha=0.9)

# ==============================================================================
# 6. BOTTOM LATEX GENERATIVE EQUATIONS BOX
# ==============================================================================
latex_text = (
    r"$\mathbf{Log\text{-}Additive\ Generative\ PSD\ Model:}\quad "
    r"\log_{10} P(f) = b - \chi \log_{10}(f) + \sum_{k=1}^{K} a_k \,\Phi_k(f)$" + "\n\n"
    r"$\mathbf{Gaussian\ Profile:}\quad "
    r"\Phi_{\mathrm{Gauss}}(f) = \exp\!\left(-\frac{(f-f_k)^2}{2\sigma_k^2}\right), \qquad "
    r"\mathbf{Pseudo\text{-}Voigt\ Profile\ (\eta = 0.5):}\quad "
    r"\Phi_{\mathrm{PV}}(f) = (1-\eta)\,\Phi_{\mathrm{Gauss}}(f) + \frac{\eta}{1 + \left(\frac{f-f_k}{\gamma_k}\right)^2}, \quad "
    r"\gamma_k = \sqrt{2\ln 2}\,\sigma_k$"
)

fig.text(
    0.5, 0.015, latex_text,
    ha="center", va="bottom",
    fontsize=8.2,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc", lw=0.8),
)

plt.show()