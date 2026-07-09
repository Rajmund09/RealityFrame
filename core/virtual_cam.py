"""
core/virtual_cam.py
───────────────────
Pipes the RealityFrame output into a system virtual webcam so apps like
Zoom, Google Meet, and OBS can use it as a camera source.

Requirements
────────────
  pip install pyvirtualcam

Windows driver (ONE of these):
  • OBS Studio  →  Tools ▸ Virtual Camera ▸ Start
  • Unity Capture (https://github.com/schellingb/UnityCapture)

If pyvirtualcam is not installed the class still loads but is a no-op.
"""

import numpy as np

try:
    import pyvirtualcam
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class VirtualCamOutput:
    """
    Thin, safe wrapper around pyvirtualcam.Camera.
    Degrades gracefully when the library or driver is absent.
    """

    def __init__(self, width: int, height: int, fps: int = 30):
        self._w       = width
        self._h       = height
        self._fps     = fps
        self._cam     = None
        self._enabled = False

        if not _AVAILABLE:
            print(
                "[VirtualCam] pyvirtualcam not installed — virtual camera disabled.\n"
                "  Fix: pip install pyvirtualcam\n"
                "  Also install OBS Studio and start its Virtual Camera."
            )

    # ── Public ────────────────────────────────────────────────────────

    def toggle(self) -> bool:
        """Toggle virtual cam on/off. Returns new enabled state."""
        if not _AVAILABLE:
            return False
        if self._enabled:
            self._stop()
        else:
            self._start()
        return self._enabled

    def send(self, frame_bgr: np.ndarray):
        """Send one BGR frame to the virtual camera (non-blocking)."""
        if not self._enabled or self._cam is None:
            return
        try:
            # pyvirtualcam expects RGB
            rgb = frame_bgr[:, :, ::-1].copy()
            self._cam.send(rgb)
            self._cam.sleep_until_next_frame()
        except Exception:
            # Driver disconnected — disable silently
            self._enabled = False
            self._cam = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def available(self) -> bool:
        return _AVAILABLE

    # ── Internal ──────────────────────────────────────────────────────

    def _start(self):
        try:
            self._cam = pyvirtualcam.Camera(
                width=self._w,
                height=self._h,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            self._enabled = True
            print(f"[VirtualCam] Streaming to: {self._cam.device}")
        except Exception as exc:
            print(f"[VirtualCam] Could not start: {exc}")
            self._enabled = False
            self._cam = None

    def _stop(self):
        if self._cam:
            try:
                self._cam.close()
            except Exception:
                pass
        self._cam     = None
        self._enabled = False
        print("[VirtualCam] Stopped.")

    def __del__(self):
        self._stop()
