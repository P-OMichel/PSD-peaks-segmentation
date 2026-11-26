import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import torch
from fooof import FOOOF

# Import your custom modules
from DL.models.models import DeformResMultiTaskUNet1D
from Inference.helpers import decode_peak_frequencies_subbin

# ==============================================================================
# 1. SETUP & MODEL LOADING
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "DL_checkpoints/Deformable_UNet(5).pth"

model = DeformResMultiTaskUNet1D(in_channels=1, base_filters=32).to(device)
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded DL weights from '{checkpoint_path}'")
else:
    print(f"Warning: Checkpoint '{checkpoint_path}' not found! Using initialized weights.")
model.eval()

# Load dataset reference grid (from benchmark npz to match UNet input dimension)
bm_data = np.load("eeg_psd_multinoise_benchmark_aligned.npz", allow_pickle=True)
f_grid = bm_data["f"].astype(np.float64)

# ==============================================================================
# 2. LOAD REAL EEG DATA
# ==============================================================================
npz_path = r"c:\Users\holcman\Desktop\code\Dataset processing\db_cases\vitaldb_case_2.npz"
data = np.load(npz_path, allow_pickle=True)
meta = json.loads(data['__metadata__'].item()) if isinstance(data['__metadata__'].item(), str) else data['__metadata__'].item()

eeg_candidates = ['BIS__EEG1_WAV', 'EEG1_WAV', 'SNUADC__EEG', 'EEG', 'Solar8000__EEG']
eeg_key = next((k for k in eeg_candidates if k in meta), None)
if not eeg_key:
    eeg_key = next(k for k, v in meta.items() if 'eeg' in k.lower() and not any(x in k.lower() for x in ['bis', 'sef', 'sr', 'emg', 'sqi']))

t = data[f"{eeg_key}_time"]
v = data[f"{eeg_key}_val"]
valid = np.isfinite(v)

t_raw = t[valid]
t_rel_sec = t_raw - t_raw[0]
t_minutes = t_rel_sec / 60.0
v_clean = v[valid] - np.median(v[valid])
fs = float(meta.get(eeg_key, {}).get('srate', 128.0))

# ==============================================================================
# 3. HELPER: RUN DL AND FOOOF ON A GIVEN REAL PSD
# ==============================================================================
def fit_methods(f_welch, pxx_welch, f_grid, model, device, center_thresh=0.35, freq_range=[0.5, 45.0]):
    # FOOOF Fit
    fm = FOOOF(
        aperiodic_mode="knee",
        peak_width_limits=[0.8, 8.0],
        max_n_peaks=4,
        min_peak_height=0.25,
        peak_threshold=2.5,
        verbose=False
    )
    fm.fit(f_welch, pxx_welch, freq_range=freq_range)
    
    fooof_peaks = fm.peak_params_[:, 0] if (fm.has_model and len(fm.peak_params_) > 0) else np.array([])
    fooof_recon_ln = fm.fooofed_spectrum_ * np.log(10.0) if fm.has_model else None
    fooof_freqs = fm.freqs if fm.has_model else None

    # DL UNet Fit
    interp_fn = interp1d(f_welch, np.log(np.maximum(pxx_welch, 1e-12)), bounds_error=False, fill_value="extrapolate")
    input_log_psd = interp_fn(f_grid).astype(np.float32)

    tensor_in = torch.tensor(input_log_psd, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        c_preds, batch_logits, _ = model(tensor_in)
        dl_curve = c_preds.squeeze().cpu().numpy()
        dl_peaks = decode_peak_frequencies_subbin(batch_logits[0, 0], f_grid, center_thresh=center_thresh)
        dl_peaks = dl_peaks[(dl_peaks >= freq_range[0]) & (dl_peaks <= freq_range[1])]

    return {
        "fooof_freqs": fooof_freqs,
        "fooof_recon": fooof_recon_ln,
        "fooof_peaks": fooof_peaks,
        "dl_curve": dl_curve,
        "dl_peaks": dl_peaks
    }

# Compute Welch PSDs for Segment 1 (9360 – 9460 s)
mask1 = (t_rel_sec >= 9360) & (t_rel_sec <= 9460)
v1 = v_clean[mask1] - np.median(v_clean[mask1])
f1_s, pxx1_s = signal.welch(v1, fs=fs, nperseg=int(fs * 2))
f1_l, pxx1_l = signal.welch(v1, fs=fs, nperseg=int(fs * 50))

# Compute Welch PSDs for Segment 2 (13500 – 13600 s)
mask2 = (t_rel_sec >= 13500) & (t_rel_sec <= 13600)
v2 = v_clean[mask2] - np.median(v_clean[mask2])
f2_s, pxx2_s = signal.welch(v2, fs=fs, nperseg=int(fs * 2))
f2_l, pxx2_l = signal.welch(v2, fs=fs, nperseg=int(fs * 50))

# Run benchmarks for all 4 combinations
res1_s = fit_methods(f1_s, pxx1_s, f_grid, model, device)
res1_l = fit_methods(f1_l, pxx1_l, f_grid, model, device)
res2_s = fit_methods(f2_s, pxx2_s, f_grid, model, device)
res2_l = fit_methods(f2_l, pxx2_l, f_grid, model, device)

# ==============================================================================
# FIGURE 1: Full Waveform & Spectrogram
# ==============================================================================
stride = max(1, len(v_clean) // 40000)
fig1, (ax_eeg, ax_spec) = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)

ax_eeg.plot(t_minutes[::stride], v_clean[::stride], lw=0.6, color='#2c3e50')
ax_eeg.set_ylim(-100, 100)
ax_eeg.set_ylabel(f"Amplitude ({meta[eeg_key].get('unit', 'uV')})")
ax_eeg.set_xlabel("Time (minutes)")
ax_eeg.set_title(f"Full EEG Waveform: {eeg_key}", fontweight="bold")
ax_eeg.grid(True, linestyle=':', alpha=0.6)

ax_eeg.axvspan(9360 / 60.0, 9460 / 60.0, color='orange', alpha=0.3, label="Segment 1 (9360-9460s)")
ax_eeg.axvspan(13500 / 60.0, 13600 / 60.0, color='purple', alpha=0.3, label="Segment 2 (13500-13600s)")
ax_eeg.legend(loc="upper right")

ax_spec.specgram(v_clean, Fs=fs, cmap='jet')
ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_xlabel("Time (seconds)")
ax_spec.set_ylim(0, 35)
ax_spec.set_title("Full Spectrogram", fontweight="bold")

# ==============================================================================
# FIGURE 2: 2x2 Grid (Top: fs*2, Bottom: fs*50)
# ==============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, constrained_layout=True)

def plot_single_psd(ax, f_w, pxx_w, res, title_str, input_label):
    # Empirical Welch PSD in log-space
    ax.plot(f_w, np.log(pxx_w), color='gray', alpha=0.55, lw=1.2, label=f"Welch Input ({input_label})")
    
    # UNet reconstructed continuous curve
    ax.plot(f_grid, res["dl_curve"], color="royalblue", lw=1.8, label="UNet Denoised")
    
    # FOOOF reconstructed continuous curve
    if res["fooof_recon"] is not None:
        ax.plot(res["fooof_freqs"], res["fooof_recon"], color="forestgreen", lw=1.6, linestyle="-.", label="FOOOF Fit")
    
    # Detected peak locations
    for idx, p in enumerate(res["dl_peaks"]):
        lbl = "UNet Peak" if idx == 0 else None
        ax.axvline(p, color="royalblue", linestyle=":", lw=1.3, alpha=0.8, label=lbl)
    for idx, p in enumerate(res["fooof_peaks"]):
        lbl = "FOOOF Peak" if idx == 0 else None
        ax.axvline(p, color="forestgreen", linestyle="--", lw=1.3, alpha=0.8, label=lbl)
        
    ax.set_xlim(0, 45)
    ax.set_ylim(-6, 4)
    ax.set_ylabel("Log Power (ln units)")
    ax.set_title(title_str, fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# Row 0 (Top): fs * 2 (Shorter window, more averaging)
plot_single_psd(axes[0, 0], f1_s, pxx1_s, res1_s, "Segment 1 (9360 – 9460 s) — nperseg = fs * 2", "fs*2")
plot_single_psd(axes[0, 1], f2_s, pxx2_s, res2_s, "Segment 2 (13500 – 13600 s) — nperseg = fs * 2", "fs*2")

# Row 1 (Bottom): fs * 50 (Longer window, fine frequency resolution)
plot_single_psd(axes[1, 0], f1_l, pxx1_l, res1_l, "Segment 1 (9360 – 9460 s) — nperseg = fs * 50", "fs*50")
plot_single_psd(axes[1, 1], f2_l, pxx2_l, res2_l, "Segment 2 (13500 – 13600 s) — nperseg = fs * 50", "fs*50")

# Common x-labels for the bottom plots
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 1].set_xlabel("Frequency (Hz)")

plt.show()