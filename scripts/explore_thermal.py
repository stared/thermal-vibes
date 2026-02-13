"""
Extract temperature arrays from Thermal Master P1 & P3 JPEG files.

Auto-detects the camera format:
- **P3** (has APP3 segments): Extracts raw uint16 thermal data, converts to Celsius.
- **P1** (no APP3): Reverse-engineers normalized (0-1) temperature from ironbow colormap.

Usage:
    uv run scripts/explore_thermal.py files/*.jpg
"""

import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# APP2 IJPEG header parsing
# ---------------------------------------------------------------------------

def parse_ijpeg_header(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Parse the APP2 IJPEG header for IR resolution info.

    Returns (ir_width, ir_height, bpp, data_size) or None if not found.
    """
    if not hasattr(img, "applist"):
        return None
    for marker, payload in img.applist:
        if marker == "APP2" and b"IJPEG" in payload[:10]:
            if len(payload) < 48:
                continue
            data_size = struct.unpack_from("<I", payload, 32)[0]
            ir_width = struct.unpack_from("<H", payload, 42)[0]
            ir_height = struct.unpack_from("<H", payload, 44)[0]
            bpp = struct.unpack_from("<H", payload, 46)[0]
            return ir_width, ir_height, bpp, data_size
    return None


# ---------------------------------------------------------------------------
# P3: raw thermal extraction
# ---------------------------------------------------------------------------

def extract_raw_thermal(img: Image.Image, ir_width: int, ir_height: int) -> np.ndarray | None:
    """Extract raw uint16 thermal frame from concatenated APP3 payloads.

    The APP3 data contains two frames stored in column-major order:
    visual/palette (first half) and raw thermal (second half).
    We auto-detect which half is thermal using byte-pair ratio, then
    reshape from column-major to row-major.

    Returns (ir_height, ir_width) uint16 array, or None if insufficient data.
    """
    if not hasattr(img, "applist"):
        return None

    app3_data = bytearray()
    for marker, payload in img.applist:
        if marker == "APP3":
            app3_data.extend(payload)

    frame_bytes = ir_width * ir_height * 2

    if len(app3_data) < frame_bytes * 2:
        return None

    first_flat = np.frombuffer(app3_data[:frame_bytes], dtype="<u2")
    second_flat = np.frombuffer(app3_data[frame_bytes:frame_bytes * 2], dtype="<u2")

    # Discriminate visual vs thermal using byte-pair ratio:
    # Visual/palette data: high_byte == low_byte (e.g. 0x4242)
    # Thermal data: high_byte != low_byte (raw sensor values)
    def paired_ratio(data: np.ndarray) -> float:
        lo = data & 0xFF
        hi = (data >> 8) & 0xFF
        return float(np.mean(lo == hi))

    first_ratio = paired_ratio(first_flat)
    second_ratio = paired_ratio(second_flat)

    # The half with LOWER paired ratio is thermal
    thermal_flat = first_flat if first_ratio < second_ratio else second_flat

    # Data is column-major. The app pre-rotates the JPEG to portrait but
    # leaves APP3 in sensor orientation. Column-major decode + rot90 CW
    # matches the JPEG portrait orientation.
    return thermal_flat.reshape(ir_width, ir_height)


def raw_to_celsius(raw: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert raw uint16 thermal values to Celsius.

    Tries candidate formulas and picks the one giving plausible indoor temperatures.
    Returns (celsius_array, formula_description).
    """
    raw_f = raw.astype(np.float64)

    formulas = [
        ("raw/16 - 273.15", raw_f / 16 - 273.15),
        ("raw/64 - 273.15 (P2)", raw_f / 64 - 273.15),
        ("raw*0.1 - 273.15 (IRG)", raw_f * 0.1 - 273.15),
    ]

    for name, temp_c in formulas:
        mean = temp_c.mean()
        if 10 < mean < 50:  # plausible indoor range
            return temp_c, name

    # Fallback: return first formula with a warning
    name, temp_c = formulas[0]
    print(f"  WARNING: No formula gave plausible indoor mean temp. Using {name} (mean={temp_c.mean():.1f}°C)")
    return temp_c, name


def parse_measurement_params(img: Image.Image) -> dict | None:
    """Parse APP5 measurement parameters (float32 values)."""
    if not hasattr(img, "applist"):
        return None

    for marker, payload in img.applist:
        if marker == "APP5" and len(payload) >= 20:
            params = {}
            fields = ["ambient_temp", "distance", "tau", "emissivity", "reflected_temp"]
            for i, name in enumerate(fields):
                offset = i * 4
                if offset + 4 <= len(payload):
                    params[name] = struct.unpack_from("<f", payload, offset)[0]
            return params
    return None


# ---------------------------------------------------------------------------
# P1: ironbow colormap reverse-engineering
# ---------------------------------------------------------------------------

def build_ironbow_lut(n: int = 256) -> np.ndarray:
    """Build a reference ironbow color palette as (n, 3) uint8 array."""
    control = np.array([
        [0.00,   0,   0,   4],
        [0.08,   0,   0,  80],
        [0.15,   0,   0, 132],
        [0.22,  20,   0, 164],
        [0.30,  64,   0, 196],
        [0.37, 112,   0, 200],
        [0.42, 152,   0, 184],
        [0.47, 188,   0, 152],
        [0.52, 216,   8, 112],
        [0.57, 236,  24,  68],
        [0.62, 248,  48,  28],
        [0.67, 255,  80,   0],
        [0.72, 255, 120,   0],
        [0.77, 255, 160,   0],
        [0.82, 255, 200,   0],
        [0.87, 255, 232,   0],
        [0.92, 255, 252,  48],
        [0.96, 255, 255, 148],
        [1.00, 255, 255, 255],
    ], dtype=np.float64)

    positions = control[:, 0]
    colors = control[:, 1:]

    lut = np.zeros((n, 3), dtype=np.uint8)
    x = np.linspace(0, 1, n)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(x, positions, colors[:, ch]), 0, 255).astype(np.uint8)
    return lut


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (uint8) array to CIE LAB. Input shape: (..., 3)."""
    rgb_f = rgb.astype(np.float64) / 255.0
    mask = rgb_f > 0.04045
    rgb_lin = np.where(mask, ((rgb_f + 0.055) / 1.055) ** 2.4, rgb_f / 12.92)

    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    z /= 1.08883

    def f(t):
        m = t > 0.008856
        return np.where(m, t ** (1 / 3), 7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ch = 200 * (fy - fz)
    return np.stack([L, a, b_ch], axis=-1)


def map_ironbow_lut(image_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map each pixel to the nearest ironbow LUT entry via 3D RGB lookup table.

    Returns normalized temperature array (0-1) at image resolution.
    """
    bin_size = 4
    n_bins = 256 // bin_size

    lut_lab = rgb_to_lab(lut)

    vals = np.arange(0, 256, bin_size, dtype=np.uint8) + bin_size // 2
    grid_r, grid_g, grid_b = np.meshgrid(vals, vals, vals, indexing="ij")
    grid_rgb = np.stack([grid_r, grid_g, grid_b], axis=-1)
    grid_lab = rgb_to_lab(grid_rgb)

    grid_flat = grid_lab.reshape(-1, 3)
    lookup_table = np.zeros(n_bins ** 3, dtype=np.uint8)
    chunk_size = 1024
    for i in range(0, len(grid_flat), chunk_size):
        chunk = grid_flat[i : i + chunk_size]
        dists = np.sum((chunk[:, None, :] - lut_lab[None, :, :]) ** 2, axis=2)
        lookup_table[i : i + chunk_size] = np.argmin(dists, axis=1)

    lookup_3d = lookup_table.reshape(n_bins, n_bins, n_bins)

    r_idx = np.clip(image_rgb[:, :, 0].astype(int) // bin_size, 0, n_bins - 1)
    g_idx = np.clip(image_rgb[:, :, 1].astype(int) // bin_size, 0, n_bins - 1)
    b_idx = np.clip(image_rgb[:, :, 2].astype(int) // bin_size, 0, n_bins - 1)

    indices = lookup_3d[r_idx, g_idx, b_idx]
    return indices.astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# Processing & visualization
# ---------------------------------------------------------------------------

def process_file(filepath: str):
    path = Path(filepath)
    img = Image.open(filepath)
    arr = np.array(img)
    h, w = arr.shape[:2]

    print(f"\n{'='*60}")
    print(f"File: {path.name} ({w}x{h})")

    # Parse IJPEG header
    header = parse_ijpeg_header(img)
    if header:
        ir_w, ir_h, bpp, data_size = header
        print(f"IJPEG header: IR {ir_w}x{ir_h} @ {bpp}bpp, data_size={data_size}")
    else:
        ir_w, ir_h = 320, 240
        print("No IJPEG header found, assuming 320x240")

    # Detect format: check for APP3 segments
    app3_count = sum(1 for m, _ in img.applist if m == "APP3") if hasattr(img, "applist") else 0

    out_dir = path.parent / "output"
    out_dir.mkdir(exist_ok=True)
    stem = path.stem

    if app3_count > 0:
        # --- P3 path: raw thermal extraction ---
        print(f"Format: P3 (found {app3_count} APP3 segments)")

        # Measurement params
        params = parse_measurement_params(img)
        if params:
            print(f"Measurement: {params}")

        raw = extract_raw_thermal(img, ir_w, ir_h)
        if raw is None:
            print("ERROR: Could not extract raw thermal data")
            return

        print(f"Raw thermal: shape={raw.shape}, min={raw.min()}, max={raw.max()}, mean={raw.mean():.1f}")

        temp_c, formula = raw_to_celsius(raw)
        print(f"Formula: {formula}")
        print(f"Temperature: min={temp_c.min():.1f}°C, max={temp_c.max():.1f}°C, "
              f"mean={temp_c.mean():.1f}°C, std={temp_c.std():.1f}°C")

        # Save .npy
        npy_path = out_dir / f"{stem}_temp_celsius.npy"
        np.save(npy_path, temp_c)
        print(f"Saved: {npy_path.name}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(arr)
        axes[0].set_title("Original (ironbow)")
        axes[0].axis("off")

        im1 = axes[1].imshow(temp_c, cmap="inferno")
        axes[1].set_title(f"Temperature {ir_w}x{ir_h}")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="°C")

        fig.suptitle(f"{path.name} — P3 raw ({formula})", fontsize=11)
        plt.tight_layout()
        fig_path = out_dir / f"{stem}_thermal.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig_path.name}")

    else:
        # --- P1 path: ironbow LUT reverse-engineering ---
        print("Format: P1 (no APP3 segments, using ironbow LUT)")

        lut = build_ironbow_lut(256)
        temp_norm = map_ironbow_lut(arr, lut)

        print(f"Normalized temp: min={temp_norm.min():.3f}, max={temp_norm.max():.3f}, "
              f"mean={temp_norm.mean():.3f}, std={temp_norm.std():.3f}")

        # Downscale to sensor resolution
        temp_small = np.array(
            Image.fromarray((temp_norm * 255).astype(np.uint8)).resize(
                (ir_w, ir_h), Image.Resampling.LANCZOS
            )
        ).astype(np.float64) / 255.0

        # Save .npy
        npy_path = out_dir / f"{stem}_temp_norm.npy"
        np.save(npy_path, temp_norm)
        npy_small = out_dir / f"{stem}_temp_{ir_w}x{ir_h}.npy"
        np.save(npy_small, temp_small)
        print(f"Saved: {npy_path.name}, {npy_small.name}")

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(arr)
        axes[0].set_title("Original (ironbow)")
        axes[0].axis("off")

        im1 = axes[1].imshow(temp_norm, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title(f"Temperature (norm) {w}x{h}")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized")

        im2 = axes[2].imshow(temp_small, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title(f"Temperature (norm) {ir_w}x{ir_h}")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Normalized")

        fig.suptitle(f"{path.name} — P1 ironbow LUT", fontsize=11)
        plt.tight_layout()
        fig_path = out_dir / f"{stem}_thermal.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig_path.name}")

    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file1.jpg> [file2.jpg ...]")
        sys.exit(1)

    for f in sys.argv[1:]:
        process_file(f)

    print("\nDone!")
