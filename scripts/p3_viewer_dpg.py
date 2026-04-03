"""Live thermal viewer for Thermal Master P3 using DearPyGui.

Usage:
    uv run scripts/p3_viewer_dpg.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import dearpygui.dearpygui as dpg
import matplotlib as mpl
import numpy as np

from p3_camera import GainMode, P3Camera, raw_to_celsius


# --- Constants ---

SENSOR_W, SENSOR_H = 256, 192
DISPLAY_SCALE = 3
DISPLAY_W, DISPLAY_H = SENSOR_W * DISPLAY_SCALE, SENSOR_H * DISPLAY_SCALE

COLORMAPS = ["inferno", "jet", "hot", "turbo", "magma", "gray"]
SCREENSHOT_DIR = Path("files/screenshots")


# --- State ---


@dataclass
class ViewerState:
    camera: P3Camera | None = None
    temp_celsius: np.ndarray | None = None
    auto_range: bool = True
    temp_min: float = 20.0
    temp_max: float = 35.0
    colormap_name: str = "inferno"
    pinned_point: tuple[int, int] | None = None  # (row, col)
    fps: float = 0.0
    frame_times: list[float] = field(default_factory=list)


STATE = ViewerState()


# --- Camera ---


def connect_camera() -> bool:
    try:
        STATE.camera = P3Camera()
        STATE.camera.connect()
        name, version = STATE.camera.init()
        print(f"Camera: {name}, FW: {version}")
        STATE.camera.start_streaming()
        return True
    except Exception as e:
        print(f"Camera connection failed: {e}")
        STATE.camera = None
        return False


# --- Rendering ---


def apply_colormap(temps: np.ndarray) -> np.ndarray:
    """Convert temperature array to RGBA float32 flat array for DPG texture."""
    t_range = STATE.temp_max - STATE.temp_min
    if t_range < 0.1:
        t_range = 0.1
    normalized = np.clip((temps - STATE.temp_min) / t_range, 0.0, 1.0)
    cmap = mpl.colormaps[STATE.colormap_name]
    rgba = cmap(normalized).astype(np.float32)  # (H, W, 4)
    return rgba.ravel()


# --- Callbacks ---


def on_colormap_change(sender, app_data):
    STATE.colormap_name = app_data


def on_temp_min_change(sender, app_data):
    STATE.temp_min = app_data
    STATE.auto_range = False
    dpg.set_value("auto_range", False)


def on_temp_max_change(sender, app_data):
    STATE.temp_max = app_data
    STATE.auto_range = False
    dpg.set_value("auto_range", False)


def on_auto_range_toggle(sender, app_data):
    STATE.auto_range = app_data


def on_emissivity_change(sender, app_data):
    if STATE.camera:
        STATE.camera.env_params.emissivity = app_data


def on_gain_change(sender, app_data):
    if STATE.camera:
        gain_map = {"High (-20 to 150C)": GainMode.HIGH, "Low (0 to 550C)": GainMode.LOW}
        mode = gain_map.get(app_data)
        if mode is not None:
            STATE.camera.set_gain_mode(mode)


def on_shutter_click():
    if STATE.camera and STATE.camera.streaming:
        STATE.camera.trigger_shutter()


def on_screenshot_click():
    if STATE.temp_celsius is None:
        return
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    np.save(SCREENSHOT_DIR / f"thermal_{ts}.npy", STATE.temp_celsius)
    # Save PNG via matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(STATE.temp_celsius, cmap=STATE.colormap_name,
                   vmin=STATE.temp_min, vmax=STATE.temp_max)
    plt.colorbar(im, ax=ax, label="°C")
    ax.set_title(f"{STATE.temp_min:.1f} – {STATE.temp_max:.1f} °C")
    fig.savefig(SCREENSHOT_DIR / f"thermal_{ts}.png", dpi=150)
    plt.close(fig)
    print(f"Saved screenshot: thermal_{ts}")


def on_image_click(sender, app_data):
    coords = get_thermal_coords()
    if coords and STATE.temp_celsius is not None:
        STATE.pinned_point = coords


def get_thermal_coords() -> tuple[int, int] | None:
    """Map current mouse position to thermal pixel (row, col)."""
    mx, my = dpg.get_mouse_pos()
    ix, iy = dpg.get_item_pos("thermal_image")
    col = int((mx - ix) / DISPLAY_SCALE)
    row = int((my - iy) / DISPLAY_SCALE)
    if 0 <= col < SENSOR_W and 0 <= row < SENSOR_H:
        return (row, col)
    return None


# --- GUI Setup ---


def setup_gui():
    dpg.create_context()
    dpg.create_viewport(title="P3 Thermal Viewer", width=1060, height=640)

    # Texture at native resolution — DPG scales on GPU
    init_data = np.zeros(SENSOR_H * SENSOR_W * 4, dtype=np.float32)
    init_data[3::4] = 1.0  # alpha = 1
    with dpg.texture_registry():
        dpg.add_raw_texture(
            SENSOR_W, SENSOR_H, init_data,
            format=dpg.mvFormat_Float_rgba,
            tag="thermal_texture",
        )

    with dpg.window(tag="main_window"):
        with dpg.group(horizontal=True):
            # --- Left: Image + overlay ---
            with dpg.child_window(width=DISPLAY_W + 16, autosize_y=True):
                dpg.add_image(
                    "thermal_texture", tag="thermal_image",
                    width=DISPLAY_W, height=DISPLAY_H,
                )
                dpg.add_text("", tag="info_text")
                dpg.add_text("", tag="hover_text")
                dpg.add_text("", tag="pinned_text")

            # --- Right: Controls ---
            with dpg.child_window(width=240, autosize_y=True):
                dpg.add_text("Colormap")
                dpg.add_radio_button(
                    COLORMAPS, default_value="inferno",
                    tag="colormap_selector",
                    callback=on_colormap_change,
                )

                dpg.add_separator()
                dpg.add_text("Temperature Range")
                dpg.add_slider_float(
                    label="Min °C", default_value=20.0,
                    min_value=-40.0, max_value=200.0,
                    tag="temp_min_slider",
                    callback=on_temp_min_change,
                )
                dpg.add_slider_float(
                    label="Max °C", default_value=35.0,
                    min_value=-40.0, max_value=200.0,
                    tag="temp_max_slider",
                    callback=on_temp_max_change,
                )
                dpg.add_checkbox(
                    label="Auto Range (1-99 percentile)",
                    default_value=True, tag="auto_range",
                    callback=on_auto_range_toggle,
                )

                dpg.add_separator()
                dpg.add_slider_float(
                    label="Emissivity", default_value=0.95,
                    min_value=0.1, max_value=1.0,
                    tag="emissivity_slider",
                    callback=on_emissivity_change,
                )

                dpg.add_separator()
                dpg.add_text("Gain Mode")
                dpg.add_radio_button(
                    ["High (-20 to 150C)", "Low (0 to 550C)"],
                    default_value="High (-20 to 150C)",
                    tag="gain_selector",
                    callback=on_gain_change,
                )
                dpg.add_button(label="Trigger Shutter (NUC)", callback=on_shutter_click)

                dpg.add_separator()
                dpg.add_text("Histogram")
                with dpg.plot(height=160, width=-1, no_mouse_pos=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="°C", tag="hist_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="", tag="hist_y"):
                        dpg.add_bar_series([], [], tag="hist_bars", weight=0.5)

                dpg.add_separator()
                dpg.add_button(label="Save Screenshot", callback=on_screenshot_click)

    # Mouse handlers
    with dpg.item_handler_registry(tag="image_handlers"):
        dpg.add_item_clicked_handler(callback=on_image_click)
    dpg.bind_item_handler_registry("thermal_image", "image_handlers")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)


# --- Main Loop ---


def main():
    if not connect_camera():
        print("No camera — exiting.")
        return

    setup_gui()

    try:
        while dpg.is_dearpygui_running():
            t0 = time.perf_counter()

            # Read frame
            if STATE.camera and STATE.camera.streaming:
                ir, thermal = STATE.camera.read_frame_both()
                if thermal is not None:
                    temps = raw_to_celsius(thermal)
                    STATE.temp_celsius = temps

                    # Auto-range
                    if STATE.auto_range:
                        STATE.temp_min = float(np.percentile(temps, 1))
                        STATE.temp_max = float(np.percentile(temps, 99))
                        dpg.set_value("temp_min_slider", STATE.temp_min)
                        dpg.set_value("temp_max_slider", STATE.temp_max)

                    # Update texture
                    rgba_flat = apply_colormap(temps)
                    dpg.set_value("thermal_texture", rgba_flat)

                    # Update histogram (every 5th frame to save CPU)
                    if int(t0 * 5) % 2 == 0:
                        hist, edges = np.histogram(temps.ravel(), bins=40,
                                                   range=(STATE.temp_min, STATE.temp_max))
                        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
                        dpg.set_value("hist_bars", [centers, hist.tolist()])
                        dpg.fit_axis_data("hist_x")
                        dpg.fit_axis_data("hist_y")

                    # Info text
                    center_t = temps[SENSOR_H // 2, SENSOR_W // 2]
                    dpg.set_value(
                        "info_text",
                        f"FPS: {STATE.fps:.0f} | "
                        f"Range: {STATE.temp_min:.1f} – {STATE.temp_max:.1f} °C | "
                        f"Center: {center_t:.1f} °C",
                    )

                    # Mouse hover
                    coords = get_thermal_coords()
                    if coords:
                        r, c = coords
                        t = temps[r, c]
                        dpg.set_value("hover_text", f"Cursor: ({c}, {r}) → {t:.2f} °C")
                    else:
                        dpg.set_value("hover_text", "")

                    # Pinned point
                    if STATE.pinned_point:
                        pr, pc = STATE.pinned_point
                        if 0 <= pr < SENSOR_H and 0 <= pc < SENSOR_W:
                            pt = temps[pr, pc]
                            dpg.set_value("pinned_text",
                                          f"Pinned: ({pc}, {pr}) → {pt:.2f} °C")

            # FPS
            dt = time.perf_counter() - t0
            STATE.frame_times.append(dt)
            if len(STATE.frame_times) > 30:
                STATE.frame_times.pop(0)
            STATE.fps = 1.0 / (sum(STATE.frame_times) / len(STATE.frame_times))

            dpg.render_dearpygui_frame()

    finally:
        if STATE.camera:
            STATE.camera.stop_streaming()
            STATE.camera.disconnect()
        dpg.destroy_context()


if __name__ == "__main__":
    main()
