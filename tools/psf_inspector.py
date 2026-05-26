"""
tools/psf_inspector.py — Interactive PSF / synthetic-image inspector for AURA.

Usage:
    # From the repo root, with the project venv activated:
    .venv/bin/python3 tools/psf_inspector.py

Layout (three panels):
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │  LEFT PANEL         │  CENTRE PANEL        │  RIGHT PANEL         │
  │  Controls           │  Star-field image    │  PSF detail          │
  │  ─────────          │  ─────────────────   │  ───────────────────  │
  │  Telescope type     │  256×256 mono image  │  15×15 kernel at     │
  │  Seeing (r0)        │  Click → sample PSF  │  clicked pixel       │
  │  PE amplitude       │  at that location    │                      │
  │  Wind sigma         │                      │  PSF cross-sections  │
  │  n_stars            │  PSF-map overlay:    │  (horizontal slice   │
  │  n_frames           │  toggle colour map   │   + vertical slice)  │
  │  [Generate]         │  showing spatial RMS │                      │
  │  [Save PNG]         │  variance            │  Zernike / seeing    │
  │                     │                      │  stats readout       │
  └─────────────────────┴──────────────────────┴──────────────────────┘

Interaction:
  - "Generate" runs the forward model with the current slider values.
  - Clicking on any pixel in the centre panel samples the local 15×15 PSF
    and displays it on the right, along with its H/V cross-sections.
  - "PSF Map" button overlays a heat-map of spatial PSF RMS over the image.
  - "Save PNG" exports the star-field + PSF-map side-by-side to a file.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Qt imports ────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
    QSizePolicy, QFileDialog, QStatusBar, QCheckBox, QSpinBox,
    QDoubleSpinBox, QSplitter, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPixmap, QImage, QCursor

# ── Matplotlib embedded in Qt ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ── AURA physics ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from aura.data.configs import (
    DatasetConfig, TelescopeConfig, AtmosphericConfig, MountConfig,
    SensorConfig, SpatialVarianceConfig, TelescopeType, NoiseMode,
)
from aura.data.psf_engine import PSFEngine


# ─────────────────────────────────────────────────────────────────────────────
# Background worker: runs the forward model off the GUI thread
# ─────────────────────────────────────────────────────────────────────────────

class GenerateWorker(QObject):
    """Runs PSFEngine.generate() in a QThread so the GUI stays responsive."""

    finished = pyqtSignal(object, object, float)   # image_np, psf_map_np, elapsed_s
    error    = pyqtSignal(str)

    def __init__(self, cfg: DatasetConfig, psf_grid_step: int, seed: int) -> None:
        super().__init__()
        self._cfg = cfg
        self._step = psf_grid_step
        self._seed = seed

    def run(self) -> None:
        try:
            engine = PSFEngine(self._cfg, psf_grid_step=self._step)
            rng = np.random.default_rng(self._seed)
            t0 = time.perf_counter()
            image, psf_map = engine.generate(rng)
            elapsed = time.perf_counter() - t0
            self.finished.emit(image, psf_map, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib canvas helpers
# ─────────────────────────────────────────────────────────────────────────────

class ImageCanvas(FigureCanvas):
    """
    Displays the 256×256 star-field image.
    Emits pixel_clicked(y, x) when the user left-clicks.
    """

    pixel_clicked = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._fig = Figure(figsize=(4, 4), dpi=100, facecolor="#1a1a2e")
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax = self._fig.add_axes([0, 0, 1, 1])
        self._ax.set_facecolor("#0d0d1a")
        self._ax.axis("off")

        self._im      = None   # main image AxesImage
        self._overlay = None   # PSF-map overlay AxesImage
        self._crosshair_h = None
        self._crosshair_v = None

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def show_image(
        self,
        image: NDArray[np.float32],
        psf_rms: Optional[NDArray[np.float32]] = None,
        show_overlay: bool = False,
    ) -> None:
        """
        Render the star-field. Optionally overlay a heatmap of spatial PSF RMS.

        Args:
            image:       (H, W) float32 in [0, 1].
            psf_rms:     (H, W) float32, RMS of PSF kernel per pixel. None → no overlay.
            show_overlay: Whether to actually render psf_rms on top.
        """
        self._ax.cla()
        self._ax.axis("off")

        # Stretch with a square-root transfer for astrophotography aesthetics
        stretched = np.sqrt(np.clip(image, 0, 1))
        self._im = self._ax.imshow(stretched, cmap="gray", origin="upper",
                                   vmin=0, vmax=1, interpolation="nearest")

        if show_overlay and psf_rms is not None:
            norm_rms = (psf_rms - psf_rms.min()) / (psf_rms.ptp() + 1e-12)
            rgba = cm.plasma(norm_rms)
            rgba[..., 3] = 0.45 * norm_rms   # alpha proportional to RMS
            self._overlay = self._ax.imshow(rgba, origin="upper",
                                             interpolation="bilinear")

        self._crosshair_h = None
        self._crosshair_v = None
        self.draw_idle()

    def mark_pixel(self, y: int, x: int) -> None:
        """Draw crosshair at (y, x) to show the selected PSF sample location."""
        H, W = self._im.get_array().shape[:2] if self._im else (256, 256)

        if self._crosshair_h is not None:
            self._crosshair_h.remove()
        if self._crosshair_v is not None:
            self._crosshair_v.remove()

        self._crosshair_h = self._ax.axhline(y, color="#00ff88", lw=0.8, alpha=0.8)
        self._crosshair_v = self._ax.axvline(x, color="#00ff88", lw=0.8, alpha=0.8)
        self._ax.plot(x, y, "+", color="#00ff88", ms=10, mew=1.5)
        self.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes != self._ax or self._im is None:
            return
        if event.button != 1:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        arr = self._im.get_array()
        H, W = arr.shape[:2]
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        self.pixel_clicked.emit(y, x)


class PSFDetailCanvas(FigureCanvas):
    """
    Right panel: shows the local 15×15 PSF kernel + H/V cross-sections.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._fig = Figure(figsize=(4, 5), dpi=95, facecolor="#1a1a2e")
        super().__init__(self._fig)
        self.setParent(parent)
        self._fig.patch.set_facecolor("#1a1a2e")

        # 3-row layout: kernel image (top), H slice (mid), V slice (bottom)
        gs = self._fig.add_gridspec(3, 1, hspace=0.45,
                                    top=0.93, bottom=0.07, left=0.12, right=0.95)
        self._ax_kernel  = self._fig.add_subplot(gs[0])
        self._ax_hslice  = self._fig.add_subplot(gs[1])
        self._ax_vslice  = self._fig.add_subplot(gs[2])

        for ax in (self._ax_kernel, self._ax_hslice, self._ax_vslice):
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#aaaacc", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        self._ax_kernel.set_title("Local PSF kernel", color="#ccccee", fontsize=8, pad=3)
        self._ax_hslice.set_title("Horizontal slice", color="#ccccee", fontsize=7, pad=2)
        self._ax_vslice.set_title("Vertical slice",   color="#ccccee", fontsize=7, pad=2)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        for ax in (self._ax_kernel, self._ax_hslice, self._ax_vslice):
            ax.cla()
            ax.set_facecolor("#0d0d1a")
        self._ax_kernel.text(0.5, 0.5, "Click image\nto inspect PSF",
                             ha="center", va="center", color="#555577",
                             fontsize=9, transform=self._ax_kernel.transAxes)
        self._ax_kernel.axis("off")
        self.draw_idle()

    def show_kernel(self, kernel: NDArray[np.float32], y: int, x: int) -> None:
        """
        Render a K×K PSF kernel sampled at pixel (y, x).

        Args:
            kernel: (K, K) float32, sums to 1.0.
            y, x:   Pixel coordinates in the source image (for title).
        """
        K = kernel.shape[0]
        cx = cy = K // 2

        # ── Kernel image ──────────────────────────────────────────────────
        self._ax_kernel.cla()
        self._ax_kernel.set_facecolor("#0d0d1a")
        vmax = kernel.max()
        self._ax_kernel.imshow(
            kernel, cmap="inferno", origin="upper",
            vmin=0, vmax=vmax, interpolation="nearest",
            extent=[-cx - 0.5, cx + 0.5, cy + 0.5, -cy - 0.5],
        )
        self._ax_kernel.axhline(0, color="#00ff88", lw=0.6, alpha=0.6)
        self._ax_kernel.axvline(0, color="#00ff88", lw=0.6, alpha=0.6)
        self._ax_kernel.set_title(f"PSF @ ({x},{y})", color="#ccccee", fontsize=8, pad=3)
        self._ax_kernel.tick_params(colors="#aaaacc", labelsize=6)
        for spine in self._ax_kernel.spines.values():
            spine.set_edgecolor("#333355")

        # ── Horizontal slice (row through centre) ─────────────────────────
        px_coords = np.arange(K) - cx
        h_slice   = kernel[cy, :]
        self._ax_hslice.cla()
        self._ax_hslice.set_facecolor("#0d0d1a")
        self._ax_hslice.fill_between(px_coords, h_slice, alpha=0.35, color="#4488ff")
        self._ax_hslice.plot(px_coords, h_slice, color="#4488ff", lw=1.2)
        self._ax_hslice.axvline(0, color="#00ff88", lw=0.6, alpha=0.5)
        self._ax_hslice.set_xlim(px_coords[0] - 0.5, px_coords[-1] + 0.5)
        self._ax_hslice.set_ylim(bottom=0)
        self._ax_hslice.set_title("Horizontal slice", color="#ccccee", fontsize=7, pad=2)
        self._ax_hslice.tick_params(colors="#aaaacc", labelsize=6)
        for spine in self._ax_hslice.spines.values():
            spine.set_edgecolor("#333355")

        # ── Vertical slice (column through centre) ────────────────────────
        v_slice = kernel[:, cx]
        self._ax_vslice.cla()
        self._ax_vslice.set_facecolor("#0d0d1a")
        self._ax_vslice.fill_between(px_coords, v_slice, alpha=0.35, color="#ff8844")
        self._ax_vslice.plot(px_coords, v_slice, color="#ff8844", lw=1.2)
        self._ax_vslice.axvline(0, color="#00ff88", lw=0.6, alpha=0.5)
        self._ax_vslice.set_xlim(px_coords[0] - 0.5, px_coords[-1] + 0.5)
        self._ax_vslice.set_ylim(bottom=0)
        self._ax_vslice.set_title("Vertical slice", color="#ccccee", fontsize=7, pad=2)
        self._ax_vslice.tick_params(colors="#aaaacc", labelsize=6)
        for spine in self._ax_vslice.spines.values():
            spine.set_edgecolor("#333355")

        self.draw_idle()

        # Return FWHM estimates for status bar
        return _estimate_fwhm(h_slice, px_coords), _estimate_fwhm(v_slice, px_coords)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_fwhm(profile: NDArray, coords: NDArray) -> float:
    """Estimate FWHM of a 1-D profile by linear interpolation at half-maximum."""
    peak = profile.max()
    if peak < 1e-12:
        return 0.0
    half = peak / 2.0
    above = profile >= half
    if not above.any():
        return 0.0
    idxs = np.where(above)[0]
    return float(coords[idxs[-1]] - coords[idxs[0]])


def _make_styled_label(text: str, bold: bool = False, color: str = "#ccccee") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {color};{'font-weight:bold;' if bold else ''}")
    return lbl


def _section_box(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setStyleSheet("""
        QGroupBox {
            color: #8888bb;
            border: 1px solid #333355;
            border-radius: 4px;
            margin-top: 8px;
            font-size: 11px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
    """)
    return box


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class PSFInspector(QMainWindow):
    """Main application window."""

    _DARK_STYLE = """
        QMainWindow, QWidget {
            background-color: #1a1a2e;
            color: #ccccee;
            font-family: "Segoe UI", "DejaVu Sans", sans-serif;
            font-size: 12px;
        }
        QPushButton {
            background-color: #2a2a4a;
            color: #aaaadd;
            border: 1px solid #444466;
            border-radius: 4px;
            padding: 5px 12px;
            min-width: 80px;
        }
        QPushButton:hover   { background-color: #3a3a6a; color: #ffffff; }
        QPushButton:pressed { background-color: #4a4aaa; }
        QPushButton:disabled{ color: #555566; border-color: #333344; }
        QPushButton#generate_btn {
            background-color: #1e4d2b;
            color: #66ff88;
            border-color: #2d7a40;
            font-weight: bold;
        }
        QPushButton#generate_btn:hover { background-color: #2a6a3a; }
        QSlider::groove:horizontal {
            background: #2a2a4a; height: 4px; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #5555aa; width: 14px; height: 14px;
            margin: -5px 0; border-radius: 7px;
        }
        QSlider::sub-page:horizontal { background: #4466cc; border-radius: 2px; }
        QComboBox {
            background-color: #2a2a4a; border: 1px solid #444466;
            border-radius: 3px; padding: 3px 8px; color: #aaaadd;
        }
        QComboBox QAbstractItemView {
            background-color: #2a2a4a; color: #ccccee; selection-background-color: #4444aa;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #2a2a4a; border: 1px solid #444466;
            border-radius: 3px; padding: 2px 6px; color: #aaaadd;
        }
        QCheckBox { color: #aaaadd; spacing: 6px; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #555577; border-radius: 2px;
            background: #2a2a4a;
        }
        QCheckBox::indicator:checked { background: #4466cc; border-color: #6688ee; }
        QStatusBar { color: #7777aa; font-size: 11px; }
        QLabel#stat_label { color: #88aaff; font-size: 11px; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AURA — PSF Inspector")
        self.setMinimumSize(1150, 700)
        self.setStyleSheet(self._DARK_STYLE)

        # ── State ──────────────────────────────────────────────────────────
        self._image:   Optional[NDArray[np.float32]] = None   # (H, W)
        self._psf_map: Optional[NDArray[np.float32]] = None   # (H, W, K²)
        self._psf_rms: Optional[NDArray[np.float32]] = None   # (H, W)
        self._worker_thread: Optional[QThread]       = None

        # ── Layout ─────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_left_panel(),   stretch=0)
        root_layout.addWidget(self._build_centre_panel(), stretch=3)
        root_layout.addWidget(self._build_right_panel(),  stretch=2)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Configure parameters and click Generate.")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(210)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # ── Title ──────────────────────────────────────────────────────────
        title = QLabel("AURA\nPSF Inspector")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#8888ff; font-size:15px; font-weight:bold; "
                            "padding: 6px 0; letter-spacing:1px;")
        layout.addWidget(title)

        # ── Telescope ──────────────────────────────────────────────────────
        tel_box = _section_box("Telescope")
        tel_lay = QVBoxLayout(tel_box)
        tel_lay.setSpacing(4)

        tel_lay.addWidget(_make_styled_label("Type"))
        self._combo_tel = QComboBox()
        self._combo_tel.addItems(["Newtonian", "Refractor", "SCT"])
        tel_lay.addWidget(self._combo_tel)

        tel_lay.addWidget(_make_styled_label("Aperture (mm)"))
        self._spin_aperture = QDoubleSpinBox()
        self._spin_aperture.setRange(50, 500)
        self._spin_aperture.setValue(200)
        self._spin_aperture.setSingleStep(10)
        tel_lay.addWidget(self._spin_aperture)

        tel_lay.addWidget(_make_styled_label("Focal length (mm)"))
        self._spin_focal = QDoubleSpinBox()
        self._spin_focal.setRange(200, 5000)
        self._spin_focal.setValue(1000)
        self._spin_focal.setSingleStep(50)
        tel_lay.addWidget(self._spin_focal)

        layout.addWidget(tel_box)

        # ── Atmosphere ─────────────────────────────────────────────────────
        atm_box = _section_box("Atmosphere")
        atm_lay = QVBoxLayout(atm_box)
        atm_lay.setSpacing(4)

        self._lbl_r0 = _make_styled_label("Seeing r₀:  0.10 m")
        atm_lay.addWidget(self._lbl_r0)
        self._sld_r0 = self._make_slider(5, 20, 10)   # 0.05 – 0.20 m × 100
        self._sld_r0.valueChanged.connect(
            lambda v: self._lbl_r0.setText(f"Seeing r₀:  {v/100:.2f} m"))
        atm_lay.addWidget(self._sld_r0)

        self._lbl_frames = _make_styled_label("Avg frames:  24")
        atm_lay.addWidget(self._lbl_frames)
        self._sld_frames = self._make_slider(4, 64, 24)
        self._sld_frames.valueChanged.connect(
            lambda v: self._lbl_frames.setText(f"Avg frames:  {v}"))
        atm_lay.addWidget(self._sld_frames)

        layout.addWidget(atm_box)

        # ── Mount ──────────────────────────────────────────────────────────
        mnt_box = _section_box("Mount / Mechanics")
        mnt_lay = QVBoxLayout(mnt_box)
        mnt_lay.setSpacing(4)

        self._lbl_pe = _make_styled_label("PE amplitude:  3.0″")
        mnt_lay.addWidget(self._lbl_pe)
        self._sld_pe = self._make_slider(0, 80, 30)    # 0 – 8.0 arcsec × 10
        self._sld_pe.valueChanged.connect(
            lambda v: self._lbl_pe.setText(f"PE amplitude:  {v/10:.1f}″"))
        mnt_lay.addWidget(self._sld_pe)

        self._lbl_wind = _make_styled_label("Wind σ:  0.5″")
        mnt_lay.addWidget(self._lbl_wind)
        self._sld_wind = self._make_slider(0, 30, 5)   # 0 – 3.0 arcsec × 10
        self._sld_wind.valueChanged.connect(
            lambda v: self._lbl_wind.setText(f"Wind σ:  {v/10:.1f}″"))
        mnt_lay.addWidget(self._sld_wind)

        layout.addWidget(mnt_box)

        # ── Scene ──────────────────────────────────────────────────────────
        scene_box = _section_box("Scene")
        scene_lay = QVBoxLayout(scene_box)
        scene_lay.setSpacing(4)

        scene_lay.addWidget(_make_styled_label("Stars per image"))
        self._spin_stars_lo = QSpinBox(); self._spin_stars_lo.setRange(5, 100)
        self._spin_stars_lo.setValue(20)
        self._spin_stars_hi = QSpinBox(); self._spin_stars_hi.setRange(5, 200)
        self._spin_stars_hi.setValue(50)
        star_row = QHBoxLayout()
        star_row.addWidget(_make_styled_label("min")); star_row.addWidget(self._spin_stars_lo)
        star_row.addWidget(_make_styled_label("max")); star_row.addWidget(self._spin_stars_hi)
        scene_lay.addLayout(star_row)

        scene_lay.addWidget(_make_styled_label("Random seed"))
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 99999)
        self._spin_seed.setValue(42)
        scene_lay.addWidget(self._spin_seed)

        layout.addWidget(scene_box)

        # ── Actions ────────────────────────────────────────────────────────
        self._btn_generate = QPushButton("⚡  Generate")
        self._btn_generate.setObjectName("generate_btn")
        self._btn_generate.clicked.connect(self._on_generate)
        layout.addWidget(self._btn_generate)

        self._btn_save = QPushButton("💾  Save PNG")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save.setEnabled(False)
        layout.addWidget(self._btn_save)

        layout.addStretch()

        # ── Stats readout ──────────────────────────────────────────────────
        self._lbl_stats = QLabel("—")
        self._lbl_stats.setObjectName("stat_label")
        self._lbl_stats.setWordWrap(True)
        self._lbl_stats.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_stats)

        return panel

    def _build_centre_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar row
        toolbar = QWidget()
        tbar_lay = QHBoxLayout(toolbar)
        tbar_lay.setContentsMargins(0, 0, 0, 0)

        self._lbl_img_title = QLabel("Star Field")
        self._lbl_img_title.setStyleSheet("color:#8888ff; font-weight:bold; font-size:13px;")
        tbar_lay.addWidget(self._lbl_img_title)
        tbar_lay.addStretch()

        self._chk_overlay = QCheckBox("PSF RMS overlay")
        self._chk_overlay.stateChanged.connect(self._on_overlay_toggled)
        tbar_lay.addWidget(self._chk_overlay)

        layout.addWidget(toolbar)

        # Image canvas
        self._img_canvas = ImageCanvas()
        self._img_canvas.pixel_clicked.connect(self._on_pixel_clicked)
        layout.addWidget(self._img_canvas)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel("PSF Detail")
        lbl.setStyleSheet("color:#8888ff; font-weight:bold; font-size:13px;")
        layout.addWidget(lbl)

        self._psf_canvas = PSFDetailCanvas()
        layout.addWidget(self._psf_canvas)

        return panel

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_slider(lo: int, hi: int, val: int) -> QSlider:
        sld = QSlider(Qt.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(val)
        sld.setTickInterval((hi - lo) // 4)
        return sld

    def _build_config(self) -> DatasetConfig:
        """Read all widget values and assemble a DatasetConfig."""
        tel_map = {
            "Newtonian": TelescopeType.NEWTONIAN,
            "Refractor": TelescopeType.REFRACTOR,
            "SCT":       TelescopeType.SCT,
        }
        tel_type = tel_map[self._combo_tel.currentText()]

        r0_val  = self._sld_r0.value() / 100.0          # → metres
        pe_max  = self._sld_pe.value()  / 10.0           # → arcsec
        pe_min  = max(0.5, pe_max * 0.5)
        wind_s  = self._sld_wind.value() / 10.0          # → arcsec
        wind_min = max(0.05, wind_s * 0.5)
        n_frames = self._sld_frames.value()

        return DatasetConfig(
            telescope=TelescopeConfig(
                telescope_type=tel_type,
                aperture_diameter_m=self._spin_aperture.value() / 1000.0,
                focal_length_m=self._spin_focal.value() / 1000.0,
                image_size_px=256,
                kernel_size_px=15,
                pupil_grid_size=128,
            ),
            atmosphere=AtmosphericConfig(
                r0_min_m=r0_val,
                r0_max_m=r0_val + 0.001,    # near-fixed r0 for deterministic preview
                n_frames=n_frames,
            ),
            mount=MountConfig(
                pe_amplitude_arcsec_range=(pe_min, pe_max) if pe_max > 0 else (0.1, 0.2),
                pe_enabled=pe_max > 0,
                wind_sigma_arcsec_range=(wind_min, wind_s) if wind_s > 0 else (0.05, 0.1),
                wind_enabled=wind_s > 0,
            ),
            sensor=SensorConfig(noise_mode=NoiseMode.FULL),
            dataset_length=1,
            n_stars_per_image_range=(
                self._spin_stars_lo.value(),
                max(self._spin_stars_lo.value(), self._spin_stars_hi.value()),
            ),
            base_seed=self._spin_seed.value(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────────────

    def _on_generate(self) -> None:
        self._btn_generate.setEnabled(False)
        self._btn_generate.setText("⏳  Generating…")
        self._status.showMessage("Generating synthetic image…")

        cfg = self._build_config()
        seed = self._spin_seed.value()
        # PSF grid step 32 → 8×8 grid over 256px image
        worker = GenerateWorker(cfg, psf_grid_step=32, seed=seed)

        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_generate_done)
        worker.error.connect(self._on_generate_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

        self._worker_thread = thread   # keep reference alive

    def _on_generate_done(
        self,
        image: NDArray[np.float32],
        psf_map: NDArray[np.float32],
        elapsed: float,
    ) -> None:
        self._image   = image
        self._psf_map = psf_map

        # Pre-compute per-pixel PSF RMS (spread measure)
        K = int(round(psf_map.shape[-1] ** 0.5))
        H, W = image.shape
        kernels = psf_map.reshape(H, W, K, K)
        cx = cy = K // 2
        coords_1d = np.arange(K, dtype=np.float32) - cx

        # Weighted standard deviation as a scalar spread measure
        yy, xx = np.meshgrid(coords_1d, coords_1d, indexing="ij")
        r2 = (xx ** 2 + yy ** 2)
        # E[r²] per pixel → RMS radius
        self._psf_rms = np.sqrt((kernels * r2[None, None, :, :]).sum(axis=(-2, -1))).astype(np.float32)

        show_overlay = self._chk_overlay.isChecked()
        self._img_canvas.show_image(image, self._psf_rms, show_overlay)
        self._psf_canvas._show_placeholder()

        self._btn_generate.setEnabled(True)
        self._btn_generate.setText("⚡  Generate")
        self._btn_save.setEnabled(True)

        rms_mean = float(self._psf_rms.mean())
        rms_edge = float(self._psf_rms[0, :].mean() + self._psf_rms[-1, :].mean() +
                         self._psf_rms[:, 0].mean() + self._psf_rms[:, -1].mean()) / 4
        self._lbl_stats.setText(
            f"PSF RMS radius\n"
            f"Centre: {rms_mean:.2f} px\n"
            f"Edge:   {rms_edge:.2f} px\n"
            f"Generated in {elapsed:.1f}s"
        )
        self._status.showMessage(
            f"Done in {elapsed:.1f}s — click any star to inspect its local PSF."
        )

    def _on_generate_error(self, msg: str) -> None:
        self._btn_generate.setEnabled(True)
        self._btn_generate.setText("⚡  Generate")
        self._status.showMessage(f"Error: {msg}")

    def _on_overlay_toggled(self) -> None:
        if self._image is None:
            return
        self._img_canvas.show_image(
            self._image, self._psf_rms, self._chk_overlay.isChecked()
        )

    def _on_pixel_clicked(self, y: int, x: int) -> None:
        if self._psf_map is None:
            return
        H, W, K2 = self._psf_map.shape
        K = int(round(K2 ** 0.5))
        kernel = self._psf_map[y, x].reshape(K, K)

        self._img_canvas.mark_pixel(y, x)
        fwhm_h, fwhm_v = self._psf_canvas.show_kernel(kernel, y, x)

        pixel_scale = 0.62  # arcsec (default — would come from config in full impl)
        self._status.showMessage(
            f"Pixel ({x}, {y})  |  PSF FWHM  H: {fwhm_h:.2f} px ({fwhm_h*pixel_scale:.2f}″)  "
            f"V: {fwhm_v:.2f} px ({fwhm_v*pixel_scale:.2f}″)  |  "
            f"Peak: {kernel.max():.5f}"
        )

    def _on_save(self) -> None:
        if self._image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", str(Path.home() / "aura_psf_inspection.png"),
            "PNG Images (*.png)"
        )
        if not path:
            return

        K2 = self._psf_map.shape[-1]
        K  = int(round(K2 ** 0.5))
        H, W = self._image.shape
        kernels = self._psf_map.reshape(H, W, K, K)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#1a1a2e")
        fig.suptitle("AURA — Synthetic Star Field + PSF Map", color="#ccccee", fontsize=13)

        axes[0].imshow(np.sqrt(np.clip(self._image, 0, 1)), cmap="gray", origin="upper")
        axes[0].set_title("Star Field (√ stretch)", color="#ccccee", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(self._psf_rms, cmap="plasma", origin="upper")
        axes[1].set_title("PSF RMS radius [px]", color="#ccccee", fontsize=10)
        axes[1].axis("off")

        # Centre PSF
        cx_img, cy_img = W // 2, H // 2
        centre_kernel = kernels[cy_img, cx_img]
        axes[2].imshow(centre_kernel, cmap="inferno", origin="upper")
        axes[2].set_title(f"Centre PSF ({cx_img},{cy_img})", color="#ccccee", fontsize=10)
        axes[2].axis("off")

        for ax in axes:
            ax.set_facecolor("#0d0d1a")

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        self._status.showMessage(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AURA PSF Inspector")

    window = PSFInspector()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
