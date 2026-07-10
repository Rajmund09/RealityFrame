"""
core/zoom.py
────────────
Maps single-hand thumb↔index pinch distance to a smooth zoom level.

Design goals (Apple/iOS feel):
  • Dead zone  — hand resting naturally stays at 1.0× (no drift)
  • Intentional — requires a deliberate, held pinch to zoom in
  • Smooth      — slow EMA so fast hand movements don't spike the zoom
  • Reversible  — opening fingers glides back to 1.0× cleanly

Tuning constants at the top of the class for easy adjustment.
"""

import math
import cv2
import numpy as np


class ZoomController:
    """
    Continuous pinch-to-zoom using MediaPipe hand landmarks.

    Usage
    ─────
      zoom = ZoomController()

      # Each frame:
      level = zoom.update(hands)           # 1.0 – MAX_ZOOM
      output = ZoomController.apply(output, level)

      # Reset:
      zoom.reset()
    """

    # ── Tuning ────────────────────────────────────────────────────────
    MAX_ZOOM      = 2.5     # maximum magnification (2.5× feels natural)
    MIN_ZOOM      = 1.0

    # Normalised pinch distance thresholds
    # norm ≈ pinch_dist / (hand_size * NORM_SCALE)
    # Large NORM_SCALE → larger denominator → smaller norm → less sensitive
    NORM_SCALE    = 1.10    # tune this to match your hand size

    DEAD_ZONE     = 0.78    # norm above this  → target = 1.0× (hand relaxed)
    PINCH_FULL    = 0.20    # norm below this  → target = MAX_ZOOM (fully pinched)

    # EMA coefficient — lower = slower/smoother (0.04–0.08 is ideal)
    _ALPHA        = 0.05

    # Minimum zoom change per frame (prevents micro-jitter at boundaries)
    _MIN_DELTA    = 0.002

    def __init__(self):
        self._zoom   = 1.0
        self._active = False

    # ── Public ────────────────────────────────────────────────────────

    def update(self, hands) -> float:
        """
        Read one hand's landmarks and return the current zoom level.
        Only responds to exactly ONE hand in frame.
        """
        if len(hands) != 1:
            self._active = False
            # Gently drift back toward 1.0× when hand leaves
            self._zoom = self._zoom * (1 - 0.03) + 1.0 * 0.03
            if abs(self._zoom - 1.0) < 0.01:
                self._zoom = 1.0
            return self._zoom

        hand    = hands[0]
        thumb   = hand.landmark[4]
        index   = hand.landmark[8]
        wrist   = hand.landmark[0]
        mid_mcp = hand.landmark[9]

        # Normalise pinch by hand size so metric is scale-invariant
        pinch     = math.hypot(thumb.x - index.x, thumb.y - index.y)
        hand_size = math.hypot(wrist.x - mid_mcp.x, wrist.y - mid_mcp.y)

        if hand_size < 1e-6:
            return self._zoom

        norm = pinch / (hand_size * self.NORM_SCALE)

        # ── Map norm → target zoom ────────────────────────────────────
        if norm >= self.DEAD_ZONE:
            # Hand relaxed / open → no zoom
            target = self.MIN_ZOOM

        elif norm <= self.PINCH_FULL:
            # Fully pinched → max zoom
            target = self.MAX_ZOOM

        else:
            # Linear interpolation between dead-zone and full-pinch
            t      = (self.DEAD_ZONE - norm) / (self.DEAD_ZONE - self.PINCH_FULL)
            target = self.MIN_ZOOM + t * (self.MAX_ZOOM - self.MIN_ZOOM)

        # ── EMA smoothing ─────────────────────────────────────────────
        new_zoom = self._zoom * (1 - self._ALPHA) + target * self._ALPHA

        # Suppress micro-jitter
        if abs(new_zoom - self._zoom) >= self._MIN_DELTA:
            self._zoom = new_zoom

        # Snap to exactly 1.0 when very close (avoids perpetual drift)
        if abs(self._zoom - 1.0) < 0.015:
            self._zoom = 1.0

        self._active = True
        return self._zoom

    def reset(self):
        """Snap zoom back to 1.0× instantly."""
        self._zoom   = 1.0
        self._active = False

    @property
    def level(self) -> float:
        return self._zoom

    @property
    def active(self) -> bool:
        return self._active

    # ── Static helper ─────────────────────────────────────────────────

    @staticmethod
    def apply(frame: np.ndarray, zoom: float) -> np.ndarray:
        """
        Centre-crop the frame by `zoom` factor then upscale back.
        zoom=1.0 → identity.   zoom=2.0 → 2× magnification.
        """
        if zoom <= 1.01:
            return frame

        h, w = frame.shape[:2]

        ch = max(1, int(h / zoom))
        cw = max(1, int(w / zoom))

        y0 = (h - ch) // 2
        x0 = (w - cw) // 2

        cropped = frame[y0: y0 + ch, x0: x0 + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
