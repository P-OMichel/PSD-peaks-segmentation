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
# 1. GENERATIVE MODELS: LOG-ADDITIVE VS. LINEAR-ADDITIVE GAUSSIAN BUMPS
# ==============================================================================
def gaussian_bump(freqs, center, height, sigma):
    """Standard Gaussian curve: height * exp(-(f - center)^2 / (2 * sigma^2))."""
    return height * np.exp(-((freqs - center) ** 2) / (2 * sigma**2))

def generate_psd_comparison(freqs, offset, exponent, peaks, mode="log"):
    """
    Generates a noiseless PSD with Gaussian periodic bumps added in either:
      - 'log' mode:    log10(P) = offset - exponent*log10(f) + sum(Gaussian)
      - 'linear' mode: P = 10^offset / f^exponent + sum(Gaussian)
    """
    aperiodic_log = offset - exponent * np.log10(freqs)
    aperiodic_lin = 10**aperiodic_log

    periodic = np.zeros_like(freqs)
    for cf, height, sigma in peaks:
        periodic += gaussian_bump(freqs, cf, height, sigma)

    if mode == "log":
        psd_log = aperiodic_log + periodic
        psd_lin = 10**psd_log
    elif mode == "linear":
        psd_lin = aperiodic_lin + periodic
        psd_log = np.log10(psd_lin)
    else:
        raise ValueError("mode must be 'log' or 'linear'")

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
# 2. GLOBAL PARAMETERS & FOOOF SETTINGS
# ==============================================================================
freqs = np.arange(0.5, 45.1, 0.2)
freq_range = [0.5, 45.0]

true_offset = 2.0
true_exponent = 3.0

# Fixed low-frequency (delta) peak
delta_peak_log = (2.5, 2.0, 0.8)
delta_peak_lin = (2.5, 100.0, 0.8)

# Example second (alpha) peak
alpha_cf = 10.0
ex_alpha_sigma = 1.5
ex_alpha_height_log = 2.0    # +2.0 log10 power above 1/f
ex_alpha_height_lin = 10.0   # +10 linear power units

fooof_kwargs = dict(
    peak_width_limits=[0.5, 18.0],  # Increased from 10.0 -> allows full linear-additive FWHM up to sigma=3.5 Hz
    max_n_peaks=6,                  # Increased from 4 -> prevents peak-budget exhaustion if a broad bump splits
    min_peak_height=0.1,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
    verbose=False,
)

# ==============================================================================
# 3. SINGLE EXAMPLE: 1 BUMP (DELTA ONLY) VS. 2 BUMPS (DELTA + ALPHA)
# ==============================================================================
# --- Log-Additive ---
psd_log_1_lin, psd_log_1_log, true_ap_log = generate_psd_comparison(
    freqs, true_offset, true_exponent, [delta_peak_log], mode="log"
)
fm_log_1, exp_log_1 = fit_single_exponent(freqs, psd_log_1_lin, fooof_kwargs, freq_range)

psd_log_2_lin, psd_log_2_log, _ = generate_psd_comparison(
    freqs, true_offset, true_exponent, [delta_peak_log, (alpha_cf, ex_alpha_height_log, ex_alpha_sigma)], mode="log"
)
fm_log_2, exp_log_2 = fit_single_exponent(freqs, psd_log_2_lin, fooof_kwargs, freq_range)

# --- Linear-Additive ---
psd_lin_1_lin, psd_lin_1_log, _ = generate_psd_comparison(
    freqs, true_offset, true_exponent, [delta_peak_lin], mode="linear"
)
fm_lin_1, exp_lin_1 = fit_single_exponent(freqs, psd_lin_1_lin, fooof_kwargs, freq_range)

psd_lin_2_lin, psd_lin_2_log, _ = generate_psd_comparison(
    freqs, true_offset, true_exponent, [delta_peak_lin, (alpha_cf, ex_alpha_height_lin, ex_alpha_sigma)], mode="linear"
)
fm_lin_2, exp_lin_2 = fit_single_exponent(freqs, psd_lin_2_lin, fooof_kwargs, freq_range)

# ==============================================================================
# 4. PARAMETER SWEEPS: 2ND PEAK WIDTH (SIGMA) & AMPLITUDE
# ==============================================================================
n_grid = 35
sigma_grid = np.linspace(0.5, 3.5, n_grid)
amp_grid_log = np.linspace(0.2, 4.0, n_grid)
amp_grid_lin = np.geomspace(0.1, 1000.0, n_grid)

# --- Sweep 1: Second Peak Width (Sigma) ---
psds_sig_log = [
    generate_psd_comparison(freqs, true_offset, true_exponent, [delta_peak_log, (alpha_cf, ex_alpha_height_log, s)], mode="log")[0]
    for s in sigma_grid
]
err_sig_log = fit_batch_exponents(freqs, psds_sig_log, fooof_kwargs, freq_range) - true_exponent

psds_sig_lin = [
    generate_psd_comparison(freqs, true_offset, true_exponent, [delta_peak_lin, (alpha_cf, ex_alpha_height_lin, s)], mode="linear")[0]
    for s in sigma_grid
]
err_sig_lin = fit_batch_exponents(freqs, psds_sig_lin, fooof_kwargs, freq_range) - true_exponent

# --- Sweep 2: Second Peak Amplitude ---
psds_amp_log = [
    generate_psd_comparison(freqs, true_offset, true_exponent, [delta_peak_log, (alpha_cf, h, ex_alpha_sigma)], mode="log")[0]
    for h in amp_grid_log
]
err_amp_log = fit_batch_exponents(freqs, psds_amp_log, fooof_kwargs, freq_range) - true_exponent

psds_amp_lin = [
    generate_psd_comparison(freqs, true_offset, true_exponent, [delta_peak_lin, (alpha_cf, h, ex_alpha_sigma)], mode="linear")[0]
    for h in amp_grid_lin
]
err_amp_lin = fit_batch_exponents(freqs, psds_amp_lin, fooof_kwargs, freq_range) - true_exponent

# ==============================================================================
# 5. VISUALIZATION (SPACIOUS LAYOUT + LATEX EQUATIONS AT BOTTOM)
# ==============================================================================
fig = plt.figure(figsize=(16, 14.5), dpi=150)

# Generous spacing between rows and columns; bottom margin reserved for LaTeX equations
gs = fig.add_gridspec(
    3, 2,
    hspace=0.52,
    wspace=0.26,
    top=0.94,
    bottom=0.14,
    left=0.07,
    right=0.96,
)

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

# --- ROW 1: LOG-ADDITIVE (1 BUMP VS. 2 BUMPS) ---
plot_example(
    fig.add_subplot(gs[0, 0]),
    psd_log_1_log, fm_log_1, exp_log_1,
    f"(A) Log-Additive: 1 Bump (Delta Only)  |  Error = {exp_log_1 - true_exponent:+.3f}",
    "seagreen",
)
plot_example(
    fig.add_subplot(gs[0, 1]),
    psd_log_2_log, fm_log_2, exp_log_2,
    f"(B) Log-Additive: 2 Bumps (Delta + Alpha)  |  Error = {exp_log_2 - true_exponent:+.3f}",
    "seagreen",
)

# --- ROW 2: LINEAR-ADDITIVE (1 BUMP VS. 2 BUMPS) ---
plot_example(
    fig.add_subplot(gs[1, 0]),
    psd_lin_1_log, fm_lin_1, exp_lin_1,
    f"(C) Linear-Additive: 1 Bump (Delta Only)  |  Error = {exp_lin_1 - true_exponent:+.3f}",
    "darkorange",
)
plot_example(
    fig.add_subplot(gs[1, 1]),
    psd_lin_2_log, fm_lin_2, exp_lin_2,
    f"(D) Linear-Additive: 2 Bumps (Delta + Alpha)  |  Error = {exp_lin_2 - true_exponent:+.3f}",
    "darkorange",
)

# --- ROW 3: ERROR SWEEPS (SIGMA & AMPLITUDE) ---
ax_sig = fig.add_subplot(gs[2, 0])
ax_sig.axhline(0, color="black", ls="--", lw=1.0, alpha=0.7)
ax_sig.plot(sigma_grid, err_sig_log, "o-", color="seagreen", markersize=3.5, lw=1.3, label="Log-Additive ($a_\\alpha = 2.0\\ \\log_{10}$)")
ax_sig.plot(sigma_grid, err_sig_lin, "s-", color="darkorange", markersize=3.5, lw=1.3, label="Linear-Additive ($A_\\alpha = 10$ linear)")
ax_sig.set_title("(E) Exponent Error vs. 2nd Peak Width ($\\sigma$)", fontweight="bold", pad=10)
ax_sig.set_xlabel("Alpha Gaussian Standard Deviation $\\sigma$ (Hz)", labelpad=4)
ax_sig.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{\\mathrm{true}}$)", labelpad=4)
ax_sig.grid(True, ls=":", alpha=0.55)
ax_sig.legend(loc="lower left", frameon=True, framealpha=0.9)

ax_amp = fig.add_subplot(gs[2, 1])
ax_amp.axhline(0, color="black", ls="--", lw=1.0, alpha=0.7)

l1 = ax_amp.plot(amp_grid_log, err_amp_log, "o-", color="seagreen", markersize=3.5, lw=1.3, label="Log-Additive (Bottom Axis: $\\log_{10}$ height)")
ax_amp.set_xlabel("Log-Additive Amplitude $a_\\alpha$ ($\\log_{10}$ Power)", color="seagreen", fontweight="bold", labelpad=4)
ax_amp.tick_params(axis="x", colors="seagreen")
ax_amp.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{\\mathrm{true}}$)", labelpad=4)
ax_amp.grid(True, ls=":", alpha=0.55)

ax_amp_top = ax_amp.twiny()
l2 = ax_amp_top.plot(amp_grid_lin, err_amp_lin, "s-", color="darkorange", markersize=3.5, lw=1.3, label="Linear-Additive (Top Axis: Linear height)")
ax_amp_top.set_xscale("log")
ax_amp_top.set_xlabel("Linear-Additive Amplitude $A_\\alpha$ (Linear Power, log scale)", color="darkorange", fontweight="bold", labelpad=6)
ax_amp_top.tick_params(axis="x", colors="darkorange")

lines = l1 + l2
ax_amp.legend(lines, [l.get_label() for l in lines], loc="lower left", frameon=True, framealpha=0.9)
ax_amp.set_title("(F) Exponent Error vs. 2nd Peak Amplitude ($\\sigma = 1.5$ Hz)", fontweight="bold", pad=34)

# ==============================================================================
# 6. BOTTOM LATEX GENERATIVE EQUATIONS BOX
# ==============================================================================
latex_text = (
    r"$\mathbf{Generative\ PSD\ Models\ (Log_{10}\ Power\ Space):}$" + "\n\n"
    r"$\mathbf{1.\ Log\text{-}Additive\ (FOOOF\ Assumption):}\quad "
    r"\log_{10} P(f) = b - \chi \log_{10}(f) + \sum_{k=1}^{K} a_k \exp\!\left(-\frac{(f - f_k)^2}{2\sigma_k^2}\right)$" + "\n\n"
    r"$\mathbf{2.\ Linear\text{-}Additive\ (Power\ Superposition):}\quad "
    r"\log_{10} P(f) = \log_{10}\!\left[\frac{10^{b}}{f^{\chi}} + \sum_{k=1}^{K} A_k \exp\!\left(-\frac{(f - f_k)^2}{2\sigma_k^2}\right)\right]$"
)

fig.text(
    0.5, 0.018, latex_text,
    ha="center", va="bottom",
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc", lw=0.8),
)

plt.show()