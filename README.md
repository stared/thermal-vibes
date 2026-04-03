# Thermal Master P1/P3 Toolkit

Tools for extracting and viewing thermal data from [Thermal Master](https://thermalmaster.com/) P1 and P3 USB thermal cameras.

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and `libusb` (for USB camera access):

```bash
brew install libusb
uv sync
```

## Scripts

### Live viewer (P3 via USB)

Connect the P3 camera via USB-C and run:

```bash
cd scripts && uv run p3_viewer_dpg.py
```

Features:
- Live thermal feed (256x192 @ ~25fps)
- Adjustable temperature range (auto or manual)
- Multiple colormaps (inferno, jet, hot, turbo, magma, gray)
- Mouse hover/click for temperature readout
- Live temperature histogram
- Gain mode, emissivity, and shutter controls
- Screenshot export (PNG + NumPy array)

### Extract from saved JPEGs

Extract temperature data from thermal JPEG files saved by the Thermal Master app:

```bash
uv run scripts/explore_thermal.py files/*.jpg
```

Auto-detects P1 vs P3 format. Outputs `.npy` arrays and visualization PNGs to `files/output/`.

## Camera driver

USB driver (`scripts/p3_camera.py`) is from [jvdillon/p3-ir-camera](https://github.com/jvdillon/p3-ir-camera) (Apache 2.0). Works on macOS and Linux.
