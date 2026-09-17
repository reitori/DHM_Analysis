import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as colors
from skimage.restoration import unwrap_phase
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.special import expit


# =====================================================
# EDIT THESE PATHS
# =====================================================

#obj_out_dir = Path("../glass_tilt/out/thirty")
#obj_mask_dir = Path("../glass_tilt/masks/radius_30")
#ref_out_dir = Path("../glass_tilt/out/ref")
#ref_mask_dir = Path("../glass_tilt/masks/ref_thirty_mask")

#obj_out_dir = Path("../glass_perp/out/twofive")
#obj_mask_dir = Path("../glass_tilt/masks/ref_forty_mask")
#ref_out_dir = Path("../glass_perp/out/ref")

obj_out_dir = Path("../last_set/out/diffuse_half_opt")
obj_mask_dir = Path("../last_set/masks/diffuse_half_opt")
ref_out_dir = Path("../last_set/out/ref_11_8")
ref_mask_dir = Path("../last_set/masks/ref_11_8")

mask_name = "plus"   # "plus" or "minus"

use_beam_mask = True

cx, cy = 1760, 1260
r = 250

ymin, ymax = cy - r, cy + r
xmin, xmax = cx - r, cx + r


# =====================================================
# LOAD DATA
# =====================================================
def load_data(out_dir, mask_dir):
    fft_viz = np.asarray(
        Image.open(out_dir / "fft_visualizer.png").convert("L"),
        dtype=np.float32
    )

    F_raw = np.load(out_dir / "raw_complex_fft.npy")
    mask  = np.load(mask_dir / f"{mask_name}_mask.npy").astype(np.float32)

    with open(mask_dir / "mask_metadata.json", "r") as f:
        metadata = json.load(f)

    if mask_name == "plus":
        cx, cy = metadata["selected_center_xy"]
    else:
        cx, cy = metadata["opposite_center_xy"]
    
    return F_raw, fft_viz, mask, cx, cy

# =====================================================
# APPLY MASK
# =====================================================

def reconstructField(F_raw, fft_viz, mask, cx, cy):
    fft_vis_masked = fft_viz * mask
    F_masked = F_raw * mask

    h, w = F_raw.shape
    ccx = w // 2
    ccy = h // 2

    shift_x = int(round(ccx - cx))
    shift_y = int(round(ccy - cy))

    F_centered = np.roll(np.roll(F_masked, shift_y, axis=0), shift_x, axis=1)

    # =====================================================
    # RECONSTRUCTION
    # =====================================================

    field = np.fft.ifft2(np.fft.ifftshift(F_centered))

    return field


def compute_corrected_phase(obj_f, ref_f, mask):
    eps = 1e-12 * np.nanmax(np.abs(ref_f)**2)

    # normalized complex transmission:
    # T = H_obj / H_ref = H_obj H_ref* / |H_ref|^2
    T = np.conjugate(obj_f * np.conj(ref_f) / (np.abs(ref_f)**2 + eps))

    return unwrap_phase(np.ma.array(np.angle(T), mask=~mask))

def compute_corrected_field(obj_f, ref_f):
    eps = 1e-12 * np.nanmax(np.abs(ref_f)**2)

    # normalized complex transmission:
    # T = H_obj / H_ref = H_obj H_ref* / |H_ref|^2
    T = np.conjugate(obj_f * np.conj(ref_f) / (np.abs(ref_f)**2 + eps))

    return T


def compute_ref_index(obj_f, ref_f, thickness, wavelength, mask, plot: bool):
    air_ref_index = 1.000293

    phase_diff = compute_corrected_phase(obj_f, ref_f, mask)
    index = (wavelength / (2 * np.pi * thickness)) * phase_diff + air_ref_index

    if plot:
        plt.figure(figsize=(8,6))
        im = plt.imshow(index, origin='lower', cmap='viridis')
        plt.colorbar(im, label='Refractive Index n')
        plt.title('Recovered Refractive Index Map')
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        plt.tight_layout()
        plt.show()

    return index

def compute_height_map(obj_f, ref_f, index, wavelength, mask):
    air_ref_index = 1.000293

    phase_diff = compute_corrected_phase(obj_f, ref_f, mask)
    return phase_diff * (wavelength / (2 * np.pi * (index - air_ref_index)))

def generate_circular_mask(arr, center_x, center_y, r):
    h, w = arr.shape
    Y, X = np.ogrid[:h, :w]

    mask = (X - center_x)**2 + (Y - center_y)**2 <= r**2
    return mask

# =====================================================
# AMPLITUDE MASK
# =====================================================

def apply_amplitude_mask(height_nm, obj_f, threshold_fraction=0.2):
    amp = np.abs(obj_f)
    threshold = threshold_fraction * np.nanmax(amp)
    amp_mask = amp > threshold
    return np.where(amp_mask, height_nm, np.nan), amp_mask

# =====================================================
# PROPAGATION / REFOCUSING
# =====================================================

from propagation import angularSpectrum

# ---- EDIT THESE ----
magnification = 0.0943711
wavelength = 532e-9       # meters
dx = 1.55e-6 / magnification
dy = 1.55e-6 / magnification
z = 0.0                   # meters; start with 0, then try +/- values
n_sample = 1.50           # approximate glass
n_medium = 1.00           # air

# ---- NEW: amplitude mask toggle ----
use_amplitude_mask = False
amplitude_threshold_fraction = 0.2

obj_field = reconstructField(*load_data(obj_out_dir, obj_mask_dir))
ref_field = reconstructField(*load_data(ref_out_dir, ref_mask_dir))

# Propagate/refocus complex field.
obj_true_field = angularSpectrum(obj_field, z, wavelength, dx, dy)
ref_true_field = angularSpectrum(ref_field, z, wavelength, dx, dy)

# -------------------------------------------------
# PHASE MAPS
# -------------------------------------------------

h, w = obj_field.shape
X, Y = np.ogrid[:h, :w]
beam_mask = X + Y >= 0

if(use_beam_mask):
    beam_mask = generate_circular_mask(obj_field, cx, cy, r)

wrapped_obj_phase = np.angle(obj_true_field)
wrapped_ref_phase = np.angle(ref_true_field)

obj_phase = unwrap_phase(np.ma.array(wrapped_obj_phase, mask=~beam_mask))
ref_phase = unwrap_phase(np.ma.array(wrapped_ref_phase, mask=~beam_mask))

obj_amp = np.abs(obj_true_field)**2
ref_amp = np.abs(ref_true_field)**2

phase_diff = compute_corrected_phase(obj_true_field, ref_true_field, beam_mask)
corrected_object = compute_corrected_field(obj_true_field, ref_true_field)

# -------------------------------------------------
# HEIGHT MAP
# -------------------------------------------------

height_map = compute_height_map(obj_true_field, ref_true_field, 1.52, wavelength, beam_mask)
height_nm = height_map * 1e9

# ---- MODIFIED: conditional amplitude mask ----
if use_amplitude_mask:
    height_nm_masked, amp_mask = apply_amplitude_mask(
        height_nm,
        obj_true_field,
        threshold_fraction=amplitude_threshold_fraction
    )
else:
    height_nm_masked = height_nm
    amp_mask = np.ones_like(height_nm, dtype=bool)

plt.figure(figsize=(8,6))

im = plt.imshow(
    height_nm_masked,
    origin='lower',
    cmap='viridis',
)

plt.colorbar(im, label='Height (nm)')
plt.title('Recovered Height Variation')
plt.xlabel('x pixel')
plt.ylabel('y pixel')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'hspace': 0.2, 'wspace': 0.5})

im0 = axes[0].imshow(obj_phase, origin='lower', cmap='RdBu')
axes[0].set_title("Object Phase")
axes[0].set_xlabel("x pixel")
axes[0].set_ylabel("y pixel")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

im1 = axes[1].imshow(ref_phase, origin='lower', cmap='RdBu')
axes[1].set_title("Reference Phase")
axes[1].set_xlabel("x pixel")
axes[1].set_ylabel("y pixel")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

im2 = axes[2].imshow(phase_diff, origin='lower', cmap='RdBu')
axes[2].set_title("True Relative Object Phase")
axes[2].set_xlabel("x pixel")
axes[2].set_ylabel("y pixel")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

plt.tight_layout()
plt.show()


rel_intensity = np.abs(np.ma.array(corrected_object, mask=~beam_mask))**2

# only trust pixels where BOTH object and reference fields are strong
valid = (
    (obj_amp > 0.2 * np.nanmax(obj_amp)) &
    (ref_amp > 0.2 * np.nanmax(ref_amp))
)

rel_intensity_masked = np.where(valid, rel_intensity, np.nan)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'hspace': 0.2, 'wspace': 0.5})

im0 = axes[0].imshow(obj_amp, origin='lower', cmap='RdBu')
axes[0].set_title("Intensity Object")
axes[0].set_xlabel("x pixel")
axes[0].set_ylabel("y pixel")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

im1 = axes[1].imshow(ref_amp, origin='lower', cmap='RdBu')
axes[1].set_title("Intensity Reference")
axes[1].set_xlabel("x pixel")
axes[1].set_ylabel("y pixel")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

im2 = axes[2].imshow(
    rel_intensity_masked,
    origin='lower',
    cmap='viridis',
    vmin=np.nanpercentile(rel_intensity_masked, 2),
    vmax=np.nanpercentile(rel_intensity_masked, 98),
)
axes[2].set_title("Intensity Relative")
axes[2].set_xlabel("x pixel")
axes[2].set_ylabel("y pixel")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

plt.tight_layout()
plt.show()

if True:
    import numpy as np
    import matplotlib.pyplot as plt

    heght_map = compute_height_map(obj_true_field, ref_true_field, 1.52, wavelength, beam_mask)
    height_nm = height_map * 1e9

    if use_amplitude_mask:
        height_nm_masked, amp_mask = apply_amplitude_mask(
            height_nm,
            obj_true_field,
            threshold_fraction=amplitude_threshold_fraction
        )
    else:
        height_nm_masked = height_nm
        amp_mask = np.ones_like(height_nm, dtype=bool)

    height_crop = height_nm_masked[ymin:ymax, xmin:xmax]

    ny, nx = height_crop.shape
    x = np.arange(xmin, xmax)
    y = np.arange(ymin, ymax)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = R <= r

    height_masked = np.where(mask, height_crop, np.nan)

    height_masked = height_masked - np.nanmedian(height_masked)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, height_masked,
        cmap='viridis',
        linewidth=0,
        antialiased=True
    )

    fig.colorbar(surf, shrink=0.65, label='Height variation (nm)')

    ax.set_title('Recovered 3D Height Map in Beam Spot')
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    ax.set_zlabel('Height variation (nm)')

    ax.view_init(elev=35, azim=225)

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# FINAL ANALYSIS TO DETERMINE QUANATITATIVE FIT TO S-CURVE
# -------------------------------------------
def sigmoid(x, a, c, x_0, y_0):
    return y_0 + a * expit(c * (x - x_0))

def sigmoid_capped(x, y_low, y_high, c, x_0):
    return y_low + (y_high - y_low) * expit(c * (x - x_0))

def chi_squared(observed, expected, sigma=1):
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    sigma = np.asarray(sigma)

    return np.sum(((observed - expected) / sigma) ** 2)

def reduced_chi_squared(observed, expected, num_params, sigma=None):
    observed = np.asarray(observed)
    expected = np.asarray(expected)

    residuals = observed - expected

    if sigma is None:
        sigma = np.nanstd(residuals)

    dof = len(observed) - num_params

    return np.sum((residuals / sigma) ** 2) / dof

def rmse(observed, expected):
    observed = np.asarray(observed)
    expected = np.asarray(expected)

    return np.sqrt(np.mean((observed - expected) ** 2))

collapsed_phase = np.ma.mean(np.ma.masked_invalid(phase_diff), axis=1)
collapsed_intensity = np.ma.mean(np.ma.masked_invalid(rel_intensity), axis=1)

req_shape = collapsed_phase.shape
x_axis = np.linspace(0, 1, req_shape[0])

valid_p = ~np.ma.getmaskarray(collapsed_phase)
valid_i = ~np.ma.getmaskarray(collapsed_intensity)

x_phase = x_axis[valid_p]
y_phase = collapsed_phase[valid_p].compressed()

x_intensity = x_axis[valid_i]
y_intensity = collapsed_intensity[valid_i].compressed()

fit_p = x_phase <= 0.70
fit_i = x_intensity <= 0.70

x_fit_p = x_phase[fit_p]
y_fit_p = y_phase[fit_p]

x_fit_i = x_intensity[fit_i]
y_fit_i = y_intensity[fit_i]

p0_p = [
    y_fit_p.max() - y_fit_p.min(),
    100,
    x_fit_p[np.argmax(np.gradient(y_fit_p))],
    y_fit_p.min()
]

p0_i = [
    max(0, np.nanmin(y_fit_i)),
    min(2, np.nanmax(y_fit_i)),
    100,
    x_fit_i[np.argmax(np.gradient(y_fit_i))]
]

phase_opt, phase_cov = curve_fit(
    sigmoid,
    x_fit_p,
    y_fit_p,
    p0=p0_p,
    bounds=(
        [-np.inf, 1, 0, -np.inf],
        [ np.inf, 10000, 1, np.inf]
    ),
    maxfev=200000
)

intens_opt, intens_cov = curve_fit(
    sigmoid_capped,
    x_fit_i,
    y_fit_i,
    p0=p0_i,
    bounds=(
        [0, 0, 1, 0],
        [2, 2, 10000, 1]
    ),
    maxfev=200000
)

fitted_sigmoid_p = sigmoid(x_fit_p, *phase_opt)
fitted_sigmoid_i = sigmoid_capped(x_fit_i, *intens_opt)

print(f"Raw chi-squared of phase map: {chi_squared(y_fit_p, fitted_sigmoid_p)}")
print(f"Raw chi-squared of intensity map: {chi_squared(y_fit_i, fitted_sigmoid_i)}")
print(f"Reduced chi-squared of phase map: {reduced_chi_squared(y_fit_p, fitted_sigmoid_p, num_params=4)}")
print(f"Reduced chi-squared of intensity map: {reduced_chi_squared(y_fit_i, fitted_sigmoid_i, num_params=4)}")
print(f"RMSE of phase map: {rmse(y_fit_p, fitted_sigmoid_p)} rad")
print(f"RMSE of intensity map: {rmse(y_fit_i, fitted_sigmoid_i)}")

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)

ax.scatter(
    x_phase,
    y_phase,
    s=12,
    alpha=0.45,
    label="Collapsed phase data"
)

ax.plot(
    x_axis,
    sigmoid(x_axis, *phase_opt),
    linewidth=2.5,
    label="Sigmoid fit"
)

ax.set_xlim(0, 1)

ymin = min(y_phase.min(), sigmoid(x_axis, *phase_opt).min())
ymax = max(y_phase.max(), sigmoid(x_axis, *phase_opt).max())

padding = 0.1 * (ymax - ymin)

ax.set_ylim(ymin - padding, ymax + padding)

ax.set_title("Collapsed Relative Phase Profile")
ax.set_xlabel("Normalized vertical position")
ax.set_ylabel("Accumulated relative phase (rad)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("collapsed_phase_sigmoid_fit.png", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)

ax.scatter(
    x_intensity,
    y_intensity,
    s=12,
    alpha=0.45,
    label="Collapsed relative intensity data"
)

ax.plot(
    x_axis,
    sigmoid_capped(x_axis, *intens_opt),
    linewidth=2.5,
    label="Sigmoid fit"
)

ax.set_xlim(0, 1)

ymin = min(y_intensity.min(), sigmoid_capped(x_axis, *intens_opt).min())
ymax = max(y_intensity.max(), sigmoid_capped(x_axis, *intens_opt).max())

padding = 0.1 * (ymax - ymin)

ax.set_ylim(ymin - padding, ymax + padding)

ax.set_title("Collapsed Relative Intensity Profile")
ax.set_xlabel("Normalized vertical position")
ax.set_ylabel(r"Relative intensity $I_{\mathrm{obj}}/I_{\mathrm{ref}}$")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("collapsed_intensity_sigmoid_fit.png", dpi=300, bbox_inches="tight")
plt.show()# -------------------------------------------------