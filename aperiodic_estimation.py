# from fooof import FOOOF, FOOOFGroup
# import matplotlib.pyplot as plt
# import numpy as np


# def generate_welch_fooof_psd(
#     freqs,
#     aperiodic_params,
#     periodic_params,
#     n_welch_segments=50,
#     rng=None,
# ):
#     """Generates a synthetic PSD with realistic Welch periodogram noise.

#     In Welch's method averaging K segments, linear PSD estimates follow:
#         P_hat(f) = P_true(f) * (chi2_{2K} / (2K))
#     """
#     offset, exponent = aperiodic_params
#     aperiodic_log = offset - np.log10(freqs**exponent)

#     total_periodic_log = np.zeros_like(freqs)
#     for center_freq, height, bandwidth in periodic_params:
#         total_periodic_log += height * np.exp(
#             -((freqs - center_freq) ** 2) / (2 * bandwidth**2)
#         )

#     true_psd_log = aperiodic_log + total_periodic_log
#     true_psd_linear = 10**true_psd_log

#     if n_welch_segments is not None and n_welch_segments > 0:
#         if rng is None:
#             rng = np.random.default_rng()
#         dof = 2 * n_welch_segments
#         chi2_multiplier = rng.chisquare(df=dof, size=freqs.shape) / dof
#         psd_linear = true_psd_linear * chi2_multiplier
#         psd_log = np.log10(psd_linear)
#     else:
#         psd_linear = true_psd_linear
#         psd_log = true_psd_log

#     return psd_linear, psd_log, aperiodic_log


# def fit_batch_exponents(freqs, spectra_linear, fooof_kwargs, freq_range):
#     """Fits a 2D array of linear PSDs with FOOOFGroup and returns estimated exponents."""
#     fg = FOOOFGroup(**fooof_kwargs)
#     fg.fit(freqs, np.asarray(spectra_linear), freq_range=freq_range)
#     return fg.get_params("aperiodic_params", "exponent")


# # ==============================================================================
# # 1. Fixed Global Parameters (0.1 to 45.0 Hz, exact 0.2 Hz step = 5s window)
# # ==============================================================================
# freqs = np.arange(0.1, 45.1, 0.2)  # 225 exact bins from 0.1 to 44.9 Hz
# freq_range = [0.1, 45.0]

# # Welch noise parameter: K = 50 segments
# n_welch_segments = 10
# n_mc_trials = 30  # Number of random Welch realizations per grid point for error bars

# # Fixed Aperiodic & Delta parameters across every sample
# fixed_offset = 2.0
# fixed_exponent = 3.0
# fixed_aperiodic = (fixed_offset, fixed_exponent)
# fixed_delta = (2.5, 4.0, 1.0)  # (center_freq=2.5 Hz, height=4.0, sigma=1.0 Hz)

# # Baseline Alpha parameters
# base_alpha_cf = 10.0
# base_alpha_height = 3.0
# base_alpha_sigma = 2.0
# base_alpha = (base_alpha_cf, base_alpha_height, base_alpha_sigma)

# # Raised min_peak_height (0.25) and max_n_peaks (4) prevent Welch noise spikes
# # from stealing peak slots away from the true Delta and Alpha bumps
# fooof_kwargs = dict(
#     peak_width_limits=[0.5, 8.0],
#     max_n_peaks=4,
#     min_peak_height=0.25,
#     peak_threshold=2.0,
#     aperiodic_mode="fixed",
#     verbose=False,
# )

# # ==============================================================================
# # 2. Single Example Fits with Two Independent Welch Noise Realizations
# # ==============================================================================
# psd1_clean_lin, _, true_ap_log = generate_welch_fooof_psd(
#     freqs, fixed_aperiodic, [fixed_delta], n_welch_segments=None
# )
# psd2_clean_lin, _, _ = generate_welch_fooof_psd(
#     freqs, fixed_aperiodic, [fixed_delta, base_alpha], n_welch_segments=None
# )

# dof = 2 * n_welch_segments

# # Realization A (seed=42)
# rng_a = np.random.default_rng(42)
# mult_a = rng_a.chisquare(df=dof, size=freqs.shape) / dof

# psd1_a_lin = psd1_clean_lin * mult_a
# psd1_a_log = np.log10(psd1_a_lin)
# fm1_a = FOOOF(**fooof_kwargs)
# fm1_a.fit(freqs, psd1_a_lin, freq_range=freq_range)
# exp1_a = fm1_a.get_params("aperiodic_params", "exponent")

# psd2_a_lin = psd2_clean_lin * mult_a
# psd2_a_log = np.log10(psd2_a_lin)
# fm2_a = FOOOF(**fooof_kwargs)
# fm2_a.fit(freqs, psd2_a_lin, freq_range=freq_range)
# exp2_a = fm2_a.get_params("aperiodic_params", "exponent")

# # Realization B (seed=99) - Second independent Welch PSD example
# rng_b = np.random.default_rng(99)
# mult_b = rng_b.chisquare(df=dof, size=freqs.shape) / dof

# psd1_b_lin = psd1_clean_lin * mult_b
# psd1_b_log = np.log10(psd1_b_lin)
# fm1_b = FOOOF(**fooof_kwargs)
# fm1_b.fit(freqs, psd1_b_lin, freq_range=freq_range)
# exp1_b = fm1_b.get_params("aperiodic_params", "exponent")

# psd2_b_lin = psd2_clean_lin * mult_b
# psd2_b_log = np.log10(psd2_b_lin)
# fm2_b = FOOOF(**fooof_kwargs)
# fm2_b.fit(freqs, psd2_b_lin, freq_range=freq_range)
# exp2_b = fm2_b.get_params("aperiodic_params", "exponent")

# # ==============================================================================
# # 3. Monte Carlo Alpha Sweeps Under Welch Noise
# # ==============================================================================
# alpha_sigma_grid = np.linspace(0.4, 3.8, 30)
# alpha_height_grid = np.linspace(0.2, 6.0, 30)
# rng_mc = np.random.default_rng(123)

# # Sweep 1: Vary Alpha Width (sigma_alpha), hold Alpha Height = 3.0
# psds_alpha_w = [
#     generate_welch_fooof_psd(
#         freqs,
#         fixed_aperiodic,
#         [fixed_delta, (base_alpha_cf, base_alpha_height, w)],
#         n_welch_segments=n_welch_segments,
#         rng=rng_mc,
#     )[0]
#     for _ in range(n_mc_trials)
#     for w in alpha_sigma_grid
# ]
# exps_alpha_w = fit_batch_exponents(freqs, psds_alpha_w, fooof_kwargs, freq_range)
# err_alpha_w = (exps_alpha_w - fixed_exponent).reshape(n_mc_trials, len(alpha_sigma_grid))
# err_w_mean = np.nanmean(err_alpha_w, axis=0)
# err_w_std = np.nanstd(err_alpha_w, axis=0)

# # Sweep 2: Vary Alpha Height (a_alpha), hold Alpha Width = 2.0
# psds_alpha_h = [
#     generate_welch_fooof_psd(
#         freqs,
#         fixed_aperiodic,
#         [fixed_delta, (base_alpha_cf, h, base_alpha_sigma)],
#         n_welch_segments=n_welch_segments,
#         rng=rng_mc,
#     )[0]
#     for _ in range(n_mc_trials)
#     for h in alpha_height_grid
# ]
# exps_alpha_h = fit_batch_exponents(freqs, psds_alpha_h, fooof_kwargs, freq_range)
# err_alpha_h = (exps_alpha_h - fixed_exponent).reshape(n_mc_trials, len(alpha_height_grid))
# err_h_mean = np.nanmean(err_alpha_h, axis=0)
# err_h_std = np.nanstd(err_alpha_h, axis=0)

# # ==============================================================================
# # 4. Visualization (Top Row: Two Welch Examples + Bar Comparison | Bottom Row: Sweeps)
# # ==============================================================================
# fig = plt.figure(figsize=(17, 9.2), dpi=120)
# gs = fig.add_gridspec(2, 6, hspace=0.36, wspace=0.55)

# # --- TOP ROW ---
# # Panel 1: Welch Realization A (Delta + Alpha)
# ax0 = fig.add_subplot(gs[0, 0:2])
# ax0.plot(freqs, psd2_a_log, color="silver", linewidth=1.1, label=f"Welch PSD A (K={n_welch_segments})")
# ax0.plot(fm2_a.freqs, fm2_a.fooofed_spectrum_, color="crimson", linestyle="--", linewidth=1.8, label="FOOOF Fit")
# ax0.plot(freqs, true_ap_log, "b--", linewidth=1.8, label=f"True $1/f$ ($\\chi={fixed_exponent}$)")
# ax0.plot(
#     fm2_a.freqs,
#     fm2_a._ap_fit,
#     color="seagreen",
#     linestyle=":",
#     linewidth=2.4,
#     label=f"Est $1/f$ ($\\hat{{\\chi}}={exp2_a:.3f}$)",
# )
# ax0.set_title("Welch Realization A (Delta + Alpha)")
# ax0.set_xlabel("Frequency (Hz)")
# ax0.set_ylabel("$\\log_{10}$ Power")
# ax0.set_xlim(0.1, 45)
# ax0.grid(True, linestyle=":", alpha=0.6)
# ax0.legend(fontsize=8.5)

# # Panel 2: Welch Realization B (Delta + Alpha)
# ax1 = fig.add_subplot(gs[0, 2:4])
# ax1.plot(freqs, psd2_b_log, color="silver", linewidth=1.1, label=f"Welch PSD B (K={n_welch_segments})")
# ax1.plot(fm2_b.freqs, fm2_b.fooofed_spectrum_, color="crimson", linestyle="--", linewidth=1.8, label="FOOOF Fit")
# ax1.plot(freqs, true_ap_log, "b--", linewidth=1.8, label=f"True $1/f$ ($\\chi={fixed_exponent}$)")
# ax1.plot(
#     fm2_b.freqs,
#     fm2_b._ap_fit,
#     color="darkorange",
#     linestyle=":",
#     linewidth=2.4,
#     label=f"Est $1/f$ ($\\hat{{\\chi}}={exp2_b:.3f}$)",
# )
# ax1.set_title("Welch Realization B (Delta + Alpha)")
# ax1.set_xlabel("Frequency (Hz)")
# ax1.set_ylabel("$\\log_{10}$ Power")
# ax1.set_xlim(0.1, 45)
# ax1.grid(True, linestyle=":", alpha=0.6)
# ax1.legend(fontsize=8.5)

# # Panel 3: Grouped Bar Comparison Across Both Realizations (Delta Only vs. Delta + Alpha)
# ax2 = fig.add_subplot(gs[0, 4:6])
# labels = ["Realization A", "Realization B"]
# x = np.arange(len(labels))
# width = 0.32

# rects_delta = ax2.bar(
#     x - width / 2, [exp1_a, exp1_b], width, label="Delta Only", color="steelblue"
# )
# rects_both = ax2.bar(
#     x + width / 2, [exp2_a, exp2_b], width, label="Delta + Alpha", color="seagreen"
# )
# ax2.axhline(fixed_exponent, color="crimson", linestyle="--", linewidth=1.4, label=f"True $\\chi = {fixed_exponent}$")

# ax2.bar_label(rects_delta, fmt="%.3f", padding=3, fontsize=9)
# ax2.bar_label(rects_both, fmt="%.3f", padding=3, fontsize=9)
# ax2.set_title("Exponent Recovery Across Welch Runs")
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.set_ylabel("Estimated Exponent ($\\hat{\\chi}$)")
# ax2.set_ylim(0, max(exp1_a, exp1_b, exp2_a, exp2_b, fixed_exponent) * 1.25)
# ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
# ax2.legend(fontsize=8.5)

# # --- BOTTOM ROW: Monte Carlo Alpha Sensitivity Sweeps ---
# ax3 = fig.add_subplot(gs[1, 0:3])
# ax3.axhline(0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
# ax3.axvline(base_alpha_sigma, color="gray", linestyle=":", label="Baseline $\\sigma_\\alpha = 2.0\\,\\mathrm{Hz}$")
# ax3.plot(alpha_sigma_grid, err_w_mean, "o-", color="tab:purple", markersize=4, label="Mean Error")
# ax3.fill_between(
#     alpha_sigma_grid,
#     err_w_mean - err_w_std,
#     err_w_mean + err_w_std,
#     color="tab:purple",
#     alpha=0.2,
#     label=f"$\\pm 1$ SD ({n_mc_trials} Welch trials)",
# )
# ax3.set_title(f"Exponent Error vs. Alpha Width ($\\sigma_\\alpha$) [Welch $K={n_welch_segments}$, Fixed $a_\\alpha=3.0$]")
# ax3.set_xlabel("Alpha Std Width $\\sigma_\\alpha$ (Hz)")
# ax3.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{true}$)")
# ax3.grid(True, linestyle=":", alpha=0.6)
# ax3.legend(fontsize=8.5)

# ax4 = fig.add_subplot(gs[1, 3:6])
# ax4.axhline(0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)
# ax4.axvline(base_alpha_height, color="gray", linestyle=":", label="Baseline $a_\\alpha = 3.0$")
# ax4.plot(alpha_height_grid, err_h_mean, "s-", color="teal", markersize=4, label="Mean Error")
# ax4.fill_between(
#     alpha_height_grid,
#     err_h_mean - err_h_std,
#     err_h_mean + err_h_std,
#     color="teal",
#     alpha=0.2,
#     label=f"$\\pm 1$ SD ({n_mc_trials} Welch trials)",
# )
# ax4.set_title(f"Exponent Error vs. Alpha Height ($a_\\alpha$) [Welch $K={n_welch_segments}$, Fixed $\\sigma_\\alpha=2.0$]")
# ax4.set_xlabel("Alpha Peak Height $a_\\alpha$ ($\\log_{10}$ Power)")
# ax4.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{true}$)")
# ax4.grid(True, linestyle=":", alpha=0.6)
# ax4.legend(fontsize=8.5)

# plt.show()




# =======================================================================================================================================
#   test 
# =======================================================================================================================================

from fooof import FOOOF, FOOOFGroup
import matplotlib.pyplot as plt
import numpy as np


def generate_welch_psd(
    freqs,
    aperiodic_params,
    periodic_params,
    peak_shape="gaussian",
    n_welch_segments=10,
    rng=None,
):
    """Generates a synthetic PSD with either Gaussian or Lorentzian periodic bumps.

    Parameters
    ----------
    periodic_params : list of tuples
        (center_freq, height_log10, width)
        - For 'gaussian': width is standard deviation sigma (Hz).
        - For 'lorentzian': width is half-width at half-maximum gamma (Hz).
    peak_shape : {'gaussian', 'lorentzian'}
        Shape of the oscillatory bumps in log10 power space.
    """
    offset, exponent = aperiodic_params
    aperiodic_log = offset - np.log10(freqs**exponent)

    total_periodic_log = np.zeros_like(freqs)
    for center_freq, height, width in periodic_params:
        if peak_shape == "gaussian":
            total_periodic_log += height * np.exp(
                -((freqs - center_freq) ** 2) / (2 * width**2)
            )
        elif peak_shape == "lorentzian":
            total_periodic_log += height / (
                1.0 + ((freqs - center_freq) / width) ** 2
            )
        else:
            raise ValueError("peak_shape must be 'gaussian' or 'lorentzian'")

    true_psd_log = aperiodic_log + total_periodic_log
    true_psd_linear = 10**true_psd_log

    if n_welch_segments is not None and n_welch_segments > 0:
        if rng is None:
            rng = np.random.default_rng()
        dof = 2 * n_welch_segments
        chi2_multiplier = rng.chisquare(df=dof, size=freqs.shape) / dof
        psd_linear = true_psd_linear * chi2_multiplier
        psd_log = np.log10(psd_linear)
    else:
        psd_linear = true_psd_linear
        psd_log = true_psd_log

    return psd_linear, psd_log, aperiodic_log


def fit_batch_exponents(freqs, spectra_linear, fooof_kwargs, freq_range):
    """Fits a 2D array of linear PSDs with FOOOFGroup and returns estimated exponents."""
    fg = FOOOFGroup(**fooof_kwargs)
    fg.fit(freqs, np.asarray(spectra_linear), freq_range=freq_range)
    return fg.get_params("aperiodic_params", "exponent")


# ==============================================================================
# 1. Fixed Global Parameters (0.1 to 45.0 Hz, 0.2 Hz resolution)
# ==============================================================================
freqs = np.arange(0.1, 45.1, 0.2)
freq_range = [0.1, 45.0]

# K = 10 Welch segments (~27.5s with 5s windows) makes periodogram variance clearly visible
n_welch_segments = 10
n_mc_trials = 30

fixed_offset = 2.0
fixed_exponent = 3.0
fixed_aperiodic = (fixed_offset, fixed_exponent)

# Note: For a Gaussian with sigma, FWHM = 2.355 * sigma.
# For a Lorentzian with HWHM gamma, FWHM = 2 * gamma.
# Using gamma = 1.177 * sigma matches their Full Width at Half Maximum (FWHM)!
fixed_delta_gauss = (2.5, 4.0, 1.0)
fixed_delta_lorentz = (2.5, 4.0, 1.0 * np.sqrt(2 * np.log(2)))  # Matched FWHM

base_alpha_cf = 10.0
base_alpha_height = 3.0
base_alpha_sigma = 2.0
base_alpha_gamma = base_alpha_sigma * np.sqrt(2 * np.log(2))  # Matched FWHM (~2.355 Hz)

fooof_kwargs = dict(
    peak_width_limits=[0.5, 8.0],
    max_n_peaks=4,
    min_peak_height=0.35,  # Adjusted for K=10 Welch noise SD (~0.14 log10)
    peak_threshold=2.0,
    aperiodic_mode="fixed",
    verbose=False,
)

# ==============================================================================
# 2. Single Example Comparison (Same Welch Noise Realization)
# ==============================================================================
rng_ex = np.random.default_rng(42)
dof = 2 * n_welch_segments
shared_welch_mult = rng_ex.chisquare(df=dof, size=freqs.shape) / dof

# Gaussian bumps + Welch noise
psd_g_clean, _, true_ap_log = generate_welch_psd(
    freqs,
    fixed_aperiodic,
    [fixed_delta_gauss, (base_alpha_cf, base_alpha_height, base_alpha_sigma)],
    peak_shape="gaussian",
    n_welch_segments=None,
)
psd_g_lin = psd_g_clean * shared_welch_mult
psd_g_log = np.log10(psd_g_lin)

fm_g = FOOOF(**fooof_kwargs)
fm_g.fit(freqs, psd_g_lin, freq_range=freq_range)
exp_g = fm_g.get_params("aperiodic_params", "exponent")

# Lorentzian bumps + identical Welch noise
psd_l_clean, _, _ = generate_welch_psd(
    freqs,
    fixed_aperiodic,
    [fixed_delta_lorentz, (base_alpha_cf, base_alpha_height, base_alpha_gamma)],
    peak_shape="lorentzian",
    n_welch_segments=None,
)
psd_l_lin = psd_l_clean * shared_welch_mult
psd_l_log = np.log10(psd_l_lin)

fm_l = FOOOF(**fooof_kwargs)
fm_l.fit(freqs, psd_l_lin, freq_range=freq_range)
exp_l = fm_l.get_params("aperiodic_params", "exponent")

# ==============================================================================
# 3. Monte Carlo Alpha Sweeps: Gaussian vs. Lorentzian Bumps
# ==============================================================================
# Sweep width using matched FWHM grid so Gaussian and Lorentzian widths are directly comparable
fwhm_grid = np.linspace(1.0, 8.0, 28)
sigma_grid = fwhm_grid / (2 * np.sqrt(2 * np.log(2)))
gamma_grid = fwhm_grid / 2.0

alpha_height_grid = np.linspace(0.5, 6.0, 28)
rng_mc = np.random.default_rng(123)

# --- Sweep 1: Alpha Width (FWHM) ---
psds_w_gauss = [
    generate_welch_psd(
        freqs,
        fixed_aperiodic,
        [fixed_delta_gauss, (base_alpha_cf, base_alpha_height, sig)],
        peak_shape="gaussian",
        n_welch_segments=n_welch_segments,
        rng=rng_mc,
    )[0]
    for _ in range(n_mc_trials)
    for sig in sigma_grid
]
err_w_g = (
    fit_batch_exponents(freqs, psds_w_gauss, fooof_kwargs, freq_range) - fixed_exponent
).reshape(n_mc_trials, len(fwhm_grid))

psds_w_lorentz = [
    generate_welch_psd(
        freqs,
        fixed_aperiodic,
        [fixed_delta_lorentz, (base_alpha_cf, base_alpha_height, gam)],
        peak_shape="lorentzian",
        n_welch_segments=n_welch_segments,
        rng=rng_mc,
    )[0]
    for _ in range(n_mc_trials)
    for gam in gamma_grid
]
err_w_l = (
    fit_batch_exponents(freqs, psds_w_lorentz, fooof_kwargs, freq_range) - fixed_exponent
).reshape(n_mc_trials, len(fwhm_grid))

# --- Sweep 2: Alpha Height ---
psds_h_gauss = [
    generate_welch_psd(
        freqs,
        fixed_aperiodic,
        [fixed_delta_gauss, (base_alpha_cf, h, base_alpha_sigma)],
        peak_shape="gaussian",
        n_welch_segments=n_welch_segments,
        rng=rng_mc,
    )[0]
    for _ in range(n_mc_trials)
    for h in alpha_height_grid
]
err_h_g = (
    fit_batch_exponents(freqs, psds_h_gauss, fooof_kwargs, freq_range) - fixed_exponent
).reshape(n_mc_trials, len(alpha_height_grid))

psds_h_lorentz = [
    generate_welch_psd(
        freqs,
        fixed_aperiodic,
        [fixed_delta_lorentz, (base_alpha_cf, h, base_alpha_gamma)],
        peak_shape="lorentzian",
        n_welch_segments=n_welch_segments,
        rng=rng_mc,
    )[0]
    for _ in range(n_mc_trials)
    for h in alpha_height_grid
]
err_h_l = (
    fit_batch_exponents(freqs, psds_h_lorentz, fooof_kwargs, freq_range) - fixed_exponent
).reshape(n_mc_trials, len(alpha_height_grid))

# ==============================================================================
# 4. Visualization (Top Row: Single Examples | Bottom Row: Error Comparison)
# ==============================================================================
fig = plt.figure(figsize=(17, 9.2), dpi=120)
gs = fig.add_gridspec(2, 6, hspace=0.36, wspace=0.55)

# --- TOP ROW ---
# Panel 1: Gaussian Bumps
ax0 = fig.add_subplot(gs[0, 0:2])
ax0.plot(freqs, psd_g_log, color="silver", linewidth=1.0, label=f"Welch PSD (K={n_welch_segments})")
ax0.plot(fm_g.freqs, fm_g.fooofed_spectrum_, color="crimson", linestyle="--", linewidth=1.8, label="FOOOF Fit")
ax0.plot(freqs, true_ap_log, "b--", linewidth=1.8, label=f"True $1/f$ ($\\chi={fixed_exponent}$)")
ax0.plot(
    fm_g.freqs,
    fm_g._ap_fit,
    color="seagreen",
    linestyle=":",
    linewidth=2.4,
    label=f"Est $1/f$ ($\\hat{{\\chi}}={exp_g:.3f}$)",
)
ax0.set_title("True Bumps: Gaussian (Matched FOOOF Model)")
ax0.set_xlabel("Frequency (Hz)")
ax0.set_ylabel("$\\log_{10}$ Power")
ax0.set_xlim(0.1, 45)
ax0.grid(True, linestyle=":", alpha=0.6)
ax0.legend(fontsize=8.5)

# Panel 2: Lorentzian Bumps
ax1 = fig.add_subplot(gs[0, 2:4])
ax1.plot(freqs, psd_l_log, color="silver", linewidth=1.0, label=f"Welch PSD (K={n_welch_segments})")
ax1.plot(fm_l.freqs, fm_l.fooofed_spectrum_, color="crimson", linestyle="--", linewidth=1.8, label="FOOOF Fit")
ax1.plot(freqs, true_ap_log, "b--", linewidth=1.8, label=f"True $1/f$ ($\\chi={fixed_exponent}$)")
ax1.plot(
    fm_l.freqs,
    fm_l._ap_fit,
    color="darkorange",
    linestyle=":",
    linewidth=2.4,
    label=f"Est $1/f$ ($\\hat{{\\chi}}={exp_l:.3f}$)",
)
ax1.set_title("True Bumps: Lorentzian (Heavy Tails)")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("$\\log_{10}$ Power")
ax1.set_xlim(0.1, 45)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(fontsize=8.5)

# Panel 3: Single-Run Bar Comparison
ax2 = fig.add_subplot(gs[0, 4:6])
conds = ["True $\\chi$", "Gaussian\nBumps", "Lorentzian\nBumps"]
vals = [fixed_exponent, exp_g, exp_l]
bars = ax2.bar(conds, vals, color=["tab:blue", "seagreen", "darkorange"], width=0.48)
ax2.axhline(fixed_exponent, color="tab:blue", linestyle="--", alpha=0.6, linewidth=1.2)
ax2.bar_label(bars, fmt="%.3f", padding=4, fontsize=9.5)
ax2.set_title("Single-Run Exponent Recovery (K=10)")
ax2.set_ylabel("Exponent ($\\chi$)")
ax2.set_ylim(0, max(vals) * 1.25)
ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

# --- BOTTOM ROW: Gaussian vs. Lorentzian Error Curves ---
# Panel 4: Error vs. Alpha FWHM
ax3 = fig.add_subplot(gs[1, 0:3])
ax3.axhline(0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)

m_wg, s_wg = np.nanmean(err_w_g, axis=0), np.nanstd(err_w_g, axis=0)
m_wl, s_wl = np.nanmean(err_w_l, axis=0), np.nanstd(err_w_l, axis=0)

ax3.plot(fwhm_grid, m_wg, "o-", color="seagreen", markersize=4, label="Gaussian Bumps (Mean)")
ax3.fill_between(fwhm_grid, m_wg - s_wg, m_wg + s_wg, color="seagreen", alpha=0.18)

ax3.plot(fwhm_grid, m_wl, "s-", color="darkorange", markersize=4, label="Lorentzian Bumps (Mean)")
ax3.fill_between(fwhm_grid, m_wl - s_wl, m_wl + s_wl, color="darkorange", alpha=0.18)

ax3.set_title(f"Exponent Error vs. Alpha FWHM [Welch $K={n_welch_segments}$, Fixed $a_\\alpha=3.0$]")
ax3.set_xlabel("Alpha Full Width at Half Maximum — FWHM (Hz)")
ax3.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{true}$)")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(fontsize=8.5)

# Panel 5: Error vs. Alpha Height
ax4 = fig.add_subplot(gs[1, 3:6])
ax4.axhline(0, color="black", linestyle="--", linewidth=1.1, alpha=0.7)

m_hg, s_hg = np.nanmean(err_h_g, axis=0), np.nanstd(err_h_g, axis=0)
m_hl, s_hl = np.nanmean(err_h_l, axis=0), np.nanstd(err_h_l, axis=0)

ax4.plot(alpha_height_grid, m_hg, "o-", color="seagreen", markersize=4, label="Gaussian Bumps (Mean)")
ax4.fill_between(alpha_height_grid, m_hg - s_hg, m_hg + s_hg, color="seagreen", alpha=0.18)

ax4.plot(alpha_height_grid, m_hl, "s-", color="darkorange", markersize=4, label="Lorentzian Bumps (Mean)")
ax4.fill_between(alpha_height_grid, m_hl - s_hl, m_hl + s_hl, color="darkorange", alpha=0.18)

ax4.set_title(f"Exponent Error vs. Alpha Height ($a_\\alpha$) [Welch $K={n_welch_segments}$, Matched FWHM]")
ax4.set_xlabel("Alpha Peak Height $a_\\alpha$ ($\\log_{10}$ Power)")
ax4.set_ylabel("Exponent Error ($\\hat{\\chi} - \\chi_{true}$)")
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.legend(fontsize=8.5)

plt.show()