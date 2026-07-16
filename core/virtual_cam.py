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
import threading
import queue
import time

try:
    import pyvirtualcam
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class VirtualCamOutput:
    """
    Thin, safe wrapper around pyvirtualcam.Camera.
    Degrades gracefully when the library or driver is absent.
    Uses a background thread to prevent blocking the main pipeline.
    """

    def __init__(self, width: int, height: int, fps: int = 30):
        self._w       = width
        self._h       = height
        self._fps     = fps
        self._cam     = None
        self._enabled = False
        
        self._frame_queue = queue.Queue(maxsize=2)
        self._running = True
        self._thread = None

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
        
        # Put the frame in the queue. If it's full, just drop the oldest.
        try:
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(frame_bgr)
            except queue.Empty:
                pass
            except queue.Full:
                pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def available(self) -> bool:
        return _AVAILABLE

    # ── Internal ──────────────────────────────────────────────────────
    
    def _worker(self):
        while self._running:
            if not self._enabled or self._cam is None:
                time.sleep(0.01)
                continue
                
            try:
                frame_bgr = self._frame_queue.get(timeout=0.1)
                rgb = frame_bgr[:, :, ::-1].copy()
                self._cam.send(rgb)
                self._cam.sleep_until_next_frame()
            except queue.Empty:
                pass
            except Exception as e:
                # Driver disconnected or error
                print(f"[VirtualCam] Error sending frame: {e}")
                self._enabled = False
                if self._cam:
                    self._cam.close()
                self._cam = None

    def _start(self):
        try:
            self._cam = pyvirtualcam.Camera(
                width=self._w,
                height=self._h,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            self._enabled = True
            
            # Start worker thread if not already running
            if self._thread is None or not self._thread.is_alive():
                self._running = True
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()
                
            print(f"[VirtualCam] Streaming to: {self._cam.device}")
        except Exception as exc:
            print(f"[VirtualCam] Could not start: {exc}")
            self._enabled = False
            self._cam = None

    def _stop(self):
        self._enabled = False
        # The worker thread will idle.
        if self._cam:
            try:
                self._cam.close()
            except Exception:
                pass
        self._cam = None
        
        # Clear the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
                
        print("[VirtualCam] Stopped.")

    def close(self):
        """Fully clean up the worker thread."""
        self._running = False
        self._stop()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __del__(self):
        self.close()
