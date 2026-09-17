#!/usr/bin/env python3
"""
dhm_named_mask_export.py

One-file DHM workflow / small library:

  1. Load a single hologram OR four acquisition states:
        ambient, left, right, total

  2. If any input is a video, process frame-by-frame.

  3. Build a VISUALIZER from the time-average of FFT magnitudes:
        < |FFT[I_t]| >_t
     and optionally display it as:
        log(1 + < |FFT[I_t]| >_t)

  4. Choose/click a +/- first-order mask from that log visualizer.

  5. Apply that mask to the ORIGINAL RAW COMPLEX FFT, not to the log image.

For still images, "original raw complex FFT" means FFT of the processed image.
For videos, it means FFT of selected processed frame(s). The mask center is chosen
using the time-averaged FFT magnitude, but reconstruction uses complex FFT data.

Examples
--------

Single image/video:
    python dhm_pipeline.py \
        --single hologram.mp4 \
        --interactive \
        --radius 45 \
        --name run1

Full subtraction:
    python dhm_pipeline.py \
        --ambient ambient.mp4 \
        --left left.mp4 \
        --right right.mp4 \
        --total total.mp4 \
        --scheme full \
        --interactive \
        --radius 45 \
        --name run1

Simple subtraction:
    python dhm_pipeline.py \
        --ambient ambient.mp4 \
        --total total.mp4 \
        --scheme simple \
        --interactive \
        --radius 45 \
        --name run1

Non-interactive after you know the lobe center:
    python dhm_pipeline.py \
        --ambient ambient.mp4 \
        --left left.mp4 \
        --right right.mp4 \
        --total total.mp4 \
        --scheme full \
        --center-x 426 \
        --center-y 280 \
        --radius 45 \
        --name run1

Interactive use
---------------
    1. The FFT visualizer appears.
    2. Use the matplotlib toolbar to zoom/pan.
    3. Press Enter.
    4. Click one first-order lobe.
    5. The opposite lobe is inferred automatically.

Important outputs
-----------------
    out/<name>/processed_mean.png
    out/<name>/fft_visualizer.png
    out/<name>/recon_input_frame.png
    out/<name>/raw_complex_fft.npy
    out/<name>/run_metadata.json
    masks/<name>/plus_mask.png
    masks/<name>/plus_mask.npy
    masks/<name>/minus_mask.png
    masks/<name>/minus_mask.npy
    masks/<name>/mask_metadata.json

The displayed/saved FFT visualizer is only for choosing the crop.
The reconstruction always uses the raw complex FFT.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}

STATE_AMBIENT = "ambient"
STATE_LEFT = "left"
STATE_RIGHT = "right"
STATE_TOTAL = "total"

SCHEME_SIMPLE = "simple"
SCHEME_FULL = "full"


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def detect_file_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    fail(f"Unsupported file extension for '{path}'")


def load_gray_image(path: str) -> np.ndarray:
    try:
        img = Image.open(path).convert("L")
    except Exception as e:
        fail(f"Could not open image '{path}': {e}")
    return np.asarray(img, dtype=np.float32)


def resize_to_match(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


class GraySource:
    """A reusable image/video source that yields grayscale float32 frames."""

    def __init__(self, path: str):
        self.path = path
        self.kind = detect_file_kind(path)
        self._image = None
        self.frame_count = None
        self.shape = self._probe_shape_and_count()

    def _probe_shape_and_count(self) -> Tuple[int, int]:
        if self.kind == "image":
            self._image = load_gray_image(self.path)
            self.frame_count = 1
            return self._image.shape

        if cv2 is None:
            fail("OpenCV/cv2 is required for video input. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            fail(f"Could not open video '{self.path}'")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            fail(f"Could not read first frame of video '{self.path}'")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count = count if count > 0 else None
        return gray.shape

    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        if self.kind == "image":
            yield self._image.astype(np.float32)
            return

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            fail(f"Could not open video '{self.path}'")
        n = 0
        while True:
            if max_frames is not None and n >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            yield gray
            n += 1
        cap.release()


def normalize_uint8(arr: np.ndarray, p_low: Optional[float] = None, p_high: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.uint8)
    if p_low is None:
        mn = float(np.nanmin(arr))
    else:
        mn = float(np.nanpercentile(arr[finite], p_low))
    if p_high is None:
        mx = float(np.nanmax(arr))
    else:
        mx = float(np.nanpercentile(arr[finite], p_high))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def save_image(path: str, arr: np.ndarray, p_low: Optional[float] = None, p_high: Optional[float] = None) -> None:
    Image.fromarray(normalize_uint8(arr, p_low=p_low, p_high=p_high)).save(path)


def complex_fft(arr: np.ndarray, remove_mean: bool = False, window: bool = False) -> np.ndarray:
    x = arr.astype(np.float32)
    if remove_mean:
        x = x - np.mean(x)
    if window:
        h, w = x.shape
        wy = np.hanning(h).astype(np.float32)
        wx = np.hanning(w).astype(np.float32)
        x = x * wy[:, None] * wx[None, :]
    return np.fft.fftshift(np.fft.fft2(x))


def fft_visualizer_from_avg_mag(avg_mag: np.ndarray, log: bool = True) -> np.ndarray:
    v = avg_mag.astype(np.float32)
    if log:
        v = np.log1p(v)
    return v


def processed_frame_from_states(frames: Dict[str, np.ndarray], scheme: str) -> np.ndarray:
    if STATE_TOTAL not in frames:
        fail("Internal error: missing total frame")
    if scheme == "single":
        return frames[STATE_TOTAL]
    if STATE_AMBIENT not in frames:
        fail("Internal error: missing ambient frame")
    if scheme == SCHEME_SIMPLE:
        return frames[STATE_TOTAL] - frames[STATE_AMBIENT]
    if STATE_LEFT not in frames or STATE_RIGHT not in frames:
        fail("Internal error: full subtraction requires left/right frames")
    return frames[STATE_TOTAL] - frames[STATE_LEFT] - frames[STATE_RIGHT] + frames[STATE_AMBIENT]


def iter_processed_frames(
    sources: Dict[str, GraySource],
    scheme: str,
    resize: bool = False,
    max_frames: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Yield processed spatial frames.

    For mixed image/video inputs, still images are reused for every video frame.
    For multiple videos, frames are zipped together and processing stops when the
    shortest video ends.
    """
    target_shape = sources[STATE_TOTAL].shape

    def maybe_resize(a: np.ndarray, state: str) -> np.ndarray:
        if a.shape == target_shape:
            return a
        if not resize:
            fail(
                f"Shape mismatch for {state}: got {a.shape}, expected {target_shape}. "
                "Use --resize to force resizing."
            )
        return resize_to_match(a, target_shape)

    image_constants = {}
    video_states = []
    for state, src in sources.items():
        if src.kind == "image":
            image_constants[state] = maybe_resize(next(src.frames()), state)
        else:
            video_states.append(state)

    if not video_states:
        yield processed_frame_from_states(image_constants, scheme)
        return

    iterators = {state: sources[state].frames(max_frames=max_frames) for state in video_states}
    n = 0
    while True:
        frames = dict(image_constants)
        for state, it in iterators.items():
            try:
                frames[state] = maybe_resize(next(it), state)
            except StopIteration:
                return
        yield processed_frame_from_states(frames, scheme)
        n += 1
        if max_frames is not None and n >= max_frames:
            return


def accumulate_fft_magnitude_and_mean(
    sources: Dict[str, GraySource],
    scheme: str,
    resize: bool = False,
    max_frames: Optional[int] = None,
    remove_mean: bool = False,
    window: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    avg_mag = None
    avg_spatial = None
    n = 0
    for img in iter_processed_frames(sources, scheme, resize=resize, max_frames=max_frames):
        F = complex_fft(img, remove_mean=remove_mean, window=window)
        mag = np.abs(F).astype(np.float64)
        if avg_mag is None:
            avg_mag = np.zeros_like(mag, dtype=np.float64)
            avg_spatial = np.zeros_like(img, dtype=np.float64)
        avg_mag += mag
        avg_spatial += img
        n += 1
    if n == 0:
        fail("No frames were processed")
    return (avg_mag / n).astype(np.float32), (avg_spatial / n).astype(np.float32), n


def choose_reconstruction_frame(
    sources: Dict[str, GraySource],
    scheme: str,
    resize: bool,
    max_frames: Optional[int],
    recon_frame: str,
) -> np.ndarray:
    frames = []
    for img in iter_processed_frames(sources, scheme, resize=resize, max_frames=max_frames):
        if recon_frame == "first":
            return img
        frames.append(img)
    if not frames:
        fail("No frames available for reconstruction")
    if recon_frame == "middle":
        return frames[len(frames) // 2]
    if recon_frame == "mean-spatial":
        return np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)
    fail(f"Unknown reconstruction frame mode: {recon_frame}")


def make_mask(shape: Tuple[int, int], cx: float, cy: float, radius: float, mask_type: str, sigma: Optional[float]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if mask_type == "hard":
        return (rr2 <= radius ** 2).astype(np.float32)
    if sigma is None:
        sigma = radius / 2.0
    mask = np.exp(-0.5 * rr2 / sigma ** 2)
    mask[rr2 > radius ** 2] = 0.0
    return mask.astype(np.float32)


def opposite_point(cx: float, cy: float, shape: Tuple[int, int]) -> Tuple[float, float]:
    h, w = shape
    ccx = w // 2
    ccy = h // 2
    return 2 * ccx - cx, 2 * ccy - cy


def ask_click_center(display: np.ndarray, p_low: float, p_high: float) -> Tuple[float, float]:
    # Match dhm_process_auto_fft_average.py display behavior exactly:
    # fft_display is already log1p(<|FFT|>) when --no-log is not set.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display, cmap="gray")
    ax.set_title("Zoom/pan first. Press Enter. Then click ONE first-order lobe.")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    state = {"ready": False, "point": None}

    def on_key(event):
        if event.key in ("enter", "return"):
            state["ready"] = True
            ax.set_title("Click the first-order lobe center now")
            fig.canvas.draw_idle()

    def on_click(event):
        if not state["ready"]:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        state["point"] = (float(event.xdata), float(event.ydata))
        ax.scatter([event.xdata], [event.ydata], marker="x", s=90)
        fig.canvas.draw_idle()
        plt.pause(0.15)
        plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    if state["point"] is None:
        fail("No click received. Use toolbar to zoom/pan, press Enter, then click the lobe.")
    return state["point"]


def reconstruct_from_mask(F_raw: np.ndarray, mask: np.ndarray, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply mask to raw complex FFT, shift chosen order to center, inverse FFT.

    Returns:
        shifted_spectrum, amplitude, wrapped_phase
    """
    h, w = F_raw.shape
    ccx = w // 2
    ccy = h // 2
    masked = F_raw * mask
    shift_y = int(round(ccy - cy))
    shift_x = int(round(ccx - cx))
    shifted = np.roll(np.roll(masked, shift_y, axis=0), shift_x, axis=1)
    field = np.fft.ifft2(np.fft.ifftshift(shifted))
    amp = np.abs(field).astype(np.float32)
    phase = np.angle(field).astype(np.float32)
    return shifted, amp, phase


def show_summary(spatial: np.ndarray, fft_display: np.ndarray, plus_mask: np.ndarray, minus_mask: np.ndarray, plus_phase: np.ndarray, args) -> None:
    if args.no_show:
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(spatial, cmap="gray")
    axes[0].set_title("Processed mean spatial")
    axes[1].imshow(fft_display, cmap="gray")
    axes[1].contour(plus_mask, levels=[0.1], colors="r", linewidths=0.7)
    axes[1].contour(minus_mask, levels=[0.1], colors="c", linewidths=0.7)
    axes[1].set_title("Log visualizer + masks")
    axes[2].imshow(plus_phase, cmap="twilight")
    axes[2].set_title("+ order wrapped phase")
    axes[3].imshow(normalize_uint8(np.abs(plus_phase)), cmap="gray")
    axes[3].set_title("Phase display check")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def save_mask_png_and_npy(mask_dir: Path, name: str, mask: np.ndarray) -> None:
    """Save mask both as viewable PNG and lossless float32 NPY."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    png = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(png).save(mask_dir / f"{name}_mask.png")
    np.save(mask_dir / f"{name}_mask.npy", mask.astype(np.float32))


def write_mask_metadata(mask_dir: Path, metadata: dict) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    with open(mask_dir / "mask_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_mask_from_png(path: str) -> np.ndarray:
    """
    Load a PNG mask as float32 in [0, 1].
    Useful in a later script when applying a saved mask to a raw complex FFT.
    """
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return arr / 255.0

def parse_args():
    p = argparse.ArgumentParser(description="Unified DHM preprocessing, FFT visualization, order masking, and reconstruction.")
    p.add_argument("--single", help="Single image/video hologram input")
    p.add_argument("--ambient", help="Ambient/background image/video")
    p.add_argument("--left", help="Left/reference-only or one-beam-blocked image/video")
    p.add_argument("--right", help="Right/object-only or other-beam-blocked image/video")
    p.add_argument("--total", help="Both-beams/total image/video")
    p.add_argument("--scheme", choices=(SCHEME_SIMPLE, SCHEME_FULL), default=SCHEME_SIMPLE)
    p.add_argument("--resize", action="store_true", help="Resize nonmatching inputs to total-frame shape")
    p.add_argument("--max-frames", type=int, default=None, help="Maximum video frames to process")
    p.add_argument("--remove-mean", action="store_true", help="Subtract each processed frame mean before FFT")
    p.add_argument("--window", action="store_true", help="Apply Hann window before FFT; useful for display, but can alter reconstruction")
    p.add_argument("--no-log", action="store_true", help="Do not log-scale FFT visualizer")
    p.add_argument("--display-p-low", type=float, default=1.0, help="Low percentile for FFT display")
    p.add_argument("--display-p-high", type=float, default=99.7, help="High percentile for FFT display")
    p.add_argument("--interactive", action="store_true", help="Interactively click one first-order lobe")
    p.add_argument("--center-x", type=float, help="Manual selected order x pixel")
    p.add_argument("--center-y", type=float, help="Manual selected order y pixel")
    p.add_argument("--radius", type=float, default=45.0, help="Mask radius in FFT pixels")
    p.add_argument("--mask", choices=("gaussian", "hard"), default="gaussian")
    p.add_argument("--sigma", type=float, default=None, help="Gaussian sigma. Default radius/2")
    p.add_argument("--recon-frame", choices=("first", "middle", "mean-spatial"), default="middle", help="Which raw complex FFT to reconstruct from after choosing mask. For videos, 'middle' often avoids startup exposure glitches.")
    p.add_argument("--name", required=True, help="Run name. Outputs go to out/<name>/ and masks/<name>/")
    p.add_argument("--out-root", default="out", help="Root folder for analysis outputs. Default: out")
    p.add_argument("--mask-root", default="masks", help="Root folder for reusable masks. Default: masks")
    p.add_argument("--save-reconstruction", action="store_true", help="Also save plus/minus amplitude and wrapped phase PNGs. Off by default to keep output folders small.")
    p.add_argument("--mask-only", action="store_true", help="Create/save masks and raw FFT, but do not calculate order reconstructions")
    p.add_argument("--no-show", action="store_true", help="Save files without showing final plots")
    args = p.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")
    if args.radius <= 0:
        p.error("--radius must be positive")
    if not args.single:
        if not args.ambient or not args.total:
            p.error("Use --single OR provide at least --ambient and --total")
        if args.scheme == SCHEME_FULL and (not args.left or not args.right):
            p.error("--scheme full requires --left and --right")
    if not args.interactive and (args.center_x is None or args.center_y is None):
        p.error("Use --interactive or provide --center-x and --center-y")
    return args


def build_sources(args) -> Tuple[Dict[str, GraySource], str]:
    if args.single:
        return {STATE_TOTAL: GraySource(args.single)}, "single"
    sources = {
        STATE_AMBIENT: GraySource(args.ambient),
        STATE_TOTAL: GraySource(args.total),
    }
    if args.scheme == SCHEME_FULL:
        sources[STATE_LEFT] = GraySource(args.left)
        sources[STATE_RIGHT] = GraySource(args.right)
    return sources, args.scheme


def main():
    args = parse_args()

    out_dir = Path(args.out_root) / args.name
    mask_dir = Path(args.mask_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    sources, scheme = build_sources(args)
    print("Inputs:")
    for state, src in sources.items():
        print(f"  {state:8s}: {src.kind:5s} shape={src.shape} frames={src.frame_count}")

    avg_mag, mean_spatial, nframes = accumulate_fft_magnitude_and_mean(
        sources,
        scheme,
        resize=args.resize,
        max_frames=args.max_frames,
        remove_mean=args.remove_mean,
        window=args.window,
    )
    print(f"Processed frames used for FFT visualizer: {nframes}")

    fft_display = fft_visualizer_from_avg_mag(avg_mag, log=not args.no_log)

    # Compact visual outputs.
    save_image(str(out_dir / "processed_mean.png"), mean_spatial)
    save_image(str(out_dir / "fft_visualizer.png"), fft_display)
    print(f"Saved: {out_dir / 'processed_mean.png'}")
    print(f"Saved: {out_dir / 'fft_visualizer.png'}")

    if args.interactive:
        cx, cy = ask_click_center(fft_display, args.display_p_low, args.display_p_high)
    else:
        cx, cy = args.center_x, args.center_y

    mx, my = opposite_point(cx, cy, mean_spatial.shape)
    print(f"Selected order center: x={cx:.2f}, y={cy:.2f}")
    print(f"Opposite order center: x={mx:.2f}, y={my:.2f}")

    plus_mask = make_mask(mean_spatial.shape, cx, cy, args.radius, args.mask, args.sigma)
    minus_mask = make_mask(mean_spatial.shape, mx, my, args.radius, args.mask, args.sigma)

    # Masks live only in masks/<name>/ so out/<name>/ stays clean.
    save_mask_png_and_npy(mask_dir, "plus", plus_mask)
    save_mask_png_and_npy(mask_dir, "minus", minus_mask)

    # Choose the spatial frame whose raw complex FFT should be reused later.
    recon_spatial = choose_reconstruction_frame(
        sources,
        scheme,
        resize=args.resize,
        max_frames=args.max_frames,
        recon_frame=args.recon_frame,
    )
    F_raw = complex_fft(recon_spatial, remove_mean=args.remove_mean, window=args.window)

    # IMPORTANT OUTPUT:
    # This preserves the full shifted complex FFT, including real/imaginary parts.
    # Later scripts can recover amplitude/phase or apply saved masks directly.
    np.save(out_dir / "raw_complex_fft.npy", F_raw)
    save_image(str(out_dir / "recon_input_frame.png"), recon_spatial)
    print(f"Saved: {out_dir / 'raw_complex_fft.npy'}")
    print(f"Saved: {out_dir / 'recon_input_frame.png'}")

    metadata = {
        "name": args.name,
        "out_dir": str(out_dir),
        "mask_dir": str(mask_dir),
        "image_shape_yx": list(mean_spatial.shape),
        "raw_complex_fft_file": str(out_dir / "raw_complex_fft.npy"),
        "raw_complex_fft_convention": "F_raw = np.fft.fftshift(np.fft.fft2(recon_spatial))",
        "recon_input_frame_file": str(out_dir / "recon_input_frame.png"),
        "mask_applies_to": "raw_complex_fft.npy with the same shifted FFT convention and same shape",
        "selected_center_xy": [float(cx), float(cy)],
        "opposite_center_xy": [float(mx), float(my)],
        "radius_px": float(args.radius),
        "mask_type": args.mask,
        "sigma_px": None if args.sigma is None else float(args.sigma),
        "display_was_log_scaled": not args.no_log,
        "remove_mean": bool(args.remove_mean),
        "window": bool(args.window),
        "scheme": scheme,
        "recon_frame": args.recon_frame,
        "frames_used_for_visualizer": int(nframes),
        "note": "Use masks on raw_complex_fft.npy, not on fft_visualizer.png. fft_visualizer.png is only for selecting the lobe."
    }
    write_mask_metadata(mask_dir, metadata)
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {mask_dir / 'mask_metadata.json'}")
    print(f"Saved: {out_dir / 'run_metadata.json'}")

    if args.mask_only:
        print("--mask-only selected; stopping after mask and raw FFT export.")
        return

    plus_shifted, plus_amp, plus_phase = reconstruct_from_mask(F_raw, plus_mask, cx, cy)
    minus_shifted, minus_amp, minus_phase = reconstruct_from_mask(F_raw, minus_mask, mx, my)

    if args.save_reconstruction:
        save_image(str(out_dir / "plus_recon_amp.png"), plus_amp)
        save_image(str(out_dir / "minus_recon_amp.png"), minus_amp)
        save_image(str(out_dir / "plus_recon_phase.png"), plus_phase, p_low=1, p_high=99)
        save_image(str(out_dir / "minus_recon_phase.png"), minus_phase, p_low=1, p_high=99)
        print(f"Saved optional reconstruction PNGs to: {out_dir}")

    show_summary(mean_spatial, fft_display, plus_mask, minus_mask, plus_phase, args)


if __name__ == "__main__":
    main()
