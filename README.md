# DHM_Analysis

Reconstruction and analysis code for an off-axis Mach–Zehnder digital holographic
microscope.

> Digital holographic microscopy (DHM) uses interference between a reference beam and
> an object beam to encode both amplitude and phase information in a single intensity
> image. [...] The experiment revolved around interferometric
> design, Fourier-order isolation, complex-field reconstruction, and qualitative
> sample validation.

Full write-up, including theory, setup and results: [`Capstone_Report.pdf`](Capstone_Report.pdf).

## Install

```bash
pip install -r requirements.txt
```

`opencv-python` is only needed for video inputs.

## Files

| File | Role |
| --- | --- |
| `mask_generator.py` | Pick a first-order sideband mask in Fourier space and export it. |
| `mask_auto.py` | Same, plus an automatic radius sweep (`--sweep-radius`). |
| `analysis.py` | Reconstruct the field, unwrap phase, build phase/height/intensity maps, fit sigmoids. |
| `propagation.py` | Angular-spectrum free-space propagation. |
| `pixel_counter.py` | Click two points in an image to measure a pixel distance / magnification. |

## Pipeline

Two stages: mask export, then analysis. Run the mask stage **twice** — once for the
object acquisition, once for a reference (sample-free) acquisition.

### 1. Export masks

With background subtraction (record the object beam alone, the reference beam alone,
and the full hologram):

```bash
python mask_generator.py \
    --ambient ambient.mp4 --left left.mp4 --right right.mp4 --total total.mp4 \
    --scheme full --interactive --radius 45 --name obj_run
```

Or from a single hologram:

```bash
python mask_generator.py --single hologram.png --interactive --radius 30 --name obj_run
```

Interactive selection: the FFT visualizer opens, zoom/pan with the matplotlib toolbar,
press Enter in the terminal, then click one first-order lobe. The conjugate lobe is
inferred. Once you know the lobe center you can skip the click with
`--center-x 426 --center-y 280`.

Use `mask_auto.py` with `--sweep-radius --r-min 15 --r-max 45 --r-step 1` to let the
radius be chosen automatically instead of guessing it.

Each run writes:

```
out/<name>/     raw_complex_fft.npy, fft_visualizer.png, processed_mean.png, run_metadata.json
masks/<name>/   plus_mask.npy, minus_mask.npy, (+ .png), mask_metadata.json
```

Repeat for the reference: `--name ref_run`.

### 2. Reconstruct and analyse

Edit the block at the top of `analysis.py`:

```python
obj_out_dir  = Path("out/obj_run")
obj_mask_dir = Path("masks/obj_run")
ref_out_dir  = Path("out/ref_run")
ref_mask_dir = Path("masks/ref_run")

mask_name = "plus"        # or "minus"
use_beam_mask = True
cx, cy = 1760, 1260       # beam centre, in pixels
r = 250                   # beam radius
```

and the physical parameters further down:

```python
magnification = 0.0943711     # from pixel_counter.py
wavelength    = 532e-9        # m
dx = dy = 1.55e-6 / magnification
z = 0.0                       # propagation distance, m (0 = no refocus)
```

Then:

```bash
python analysis.py
```

It masks and re-centres the first order, inverse-transforms to the complex field,
optionally propagates by `z`, unwraps the object and reference phase, and divides out
the reference to get the corrected phase difference, height map and relative intensity.
Maps are displayed interactively; the two sigmoid fits are also saved as
`collapsed_phase_sigmoid_fit.png` and `collapsed_intensity_sigmoid_fit.png`.

To get `magnification`, image a beam spot of known size and run:

```bash
python pixel_counter.py beamspot.png
```
