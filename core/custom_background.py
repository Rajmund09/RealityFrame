"""
core/custom_background.py
─────────────────────────
Loads a static image (JPG/PNG) or looping video (MP4/AVI/MOV) as the
replacement background for all invisibility effects.

Usage
─────
  bg = CustomBackground()
  ok = bg.set_source("/path/to/forest.jpg")   # or .mp4
  if ok:
      bg.enable()

  # Each frame:
  custom = bg.get_frame(frame.shape)          # BGR ndarray | None
  if custom is not None:
      background = custom                     # swap in place of captured BG
"""

import os
import cv2
import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class CustomBackground:
    """
    Provides a custom replacement background frame on demand.
    Supports still images (resized) and looping videos.
    Falls back to None when disabled or source is invalid.
    """

    def __init__(self):
        self._path    = None        # absolute path to source
        self._mode    = None        # "image" | "video"
        self._static  = None        # np.ndarray for images
        self._cap     = None        # cv2.VideoCapture for videos
        self._enabled = False

    # ── Public API ────────────────────────────────────────────────────

    def set_source(self, path: str) -> bool:
        """
        Load a new background source.
        Returns True on success, False on any error.
        """
        path = path.strip().strip('"').strip("'")   # handle copy-pasted paths
        self._cleanup()

        if not os.path.isfile(path):
            print(f"[CustomBG] File not found: {path}")
            return False

        ext = os.path.splitext(path)[1].lower()

        if ext in _IMAGE_EXTS:
            return self._load_image(path)
        elif ext in _VIDEO_EXTS:
            return self._load_video(path)
        else:
            print(f"[CustomBG] Unsupported extension: {ext}")
            return False

    def get_frame(self, target_shape) -> "np.ndarray | None":
        """
        Return a BGR frame resized to target_shape[:2] (H, W).
        Returns None when disabled or not loaded.
        """
        if not self._enabled:
            return None

        h, w = target_shape[:2]

        if self._mode == "image" and self._static is not None:
            return cv2.resize(self._static, (w, h),
                              interpolation=cv2.INTER_LINEAR)

        if self._mode == "video" and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                # Loop: rewind to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            if ret:
                return cv2.resize(frame, (w, h),
                                  interpolation=cv2.INTER_LINEAR)

        return None

    def toggle(self) -> bool:
        """Toggle on/off. Returns new enabled state."""
        if self._path is None:
            return False
        self._enabled = not self._enabled
        return self._enabled

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def source_name(self) -> "str | None":
        """Short filename for HUD display."""
        if self._path:
            name = os.path.basename(self._path)
            return name[:20] + "…" if len(name) > 20 else name
        return None

    # ── Internal ──────────────────────────────────────────────────────

    def _load_image(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is None:
            print(f"[CustomBG] Cannot decode image: {path}")
            return False
        self._static  = img
        self._mode    = "image"
        self._path    = path
        self._enabled = True
        print(f"[CustomBG] Image loaded: {os.path.basename(path)}")
        return True

    def _load_video(self, path: str) -> bool:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[CustomBG] Cannot open video: {path}")
            return False
        self._cap     = cap
        self._mode    = "video"
        self._path    = path
        self._enabled = True
        print(f"[CustomBG] Video loaded: {os.path.basename(path)}")
        return True

    def _cleanup(self):
        if self._cap:
            self._cap.release()
            self._cap = None
        self._static  = None
        self._mode    = None
        self._path    = None
        self._enabled = False

    def __del__(self):
        self._cleanup()
