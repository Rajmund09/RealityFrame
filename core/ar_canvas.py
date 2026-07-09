"""
core/ar_canvas.py
─────────────────
AR drawing canvas with body-anchored sticker support.

Drawing flow
────────────
1. Enter draw mode  (key D)
2. Point index finger → pen draws at fingertip
3. Make a fist       → pen lifts (new stroke starts on next point)
4. Press N           → anchor current drawing to nearest body landmark
5. Press U           → undo last stroke
6. Press X           → clear all free strokes

Cycle colour  → key C  (in draw mode)
Cycle brush   → key W  (in draw mode)

Body anchoring
──────────────
Each anchored sticker stores strokes as local offsets from the primary
landmark.  On render it applies translate + uniform scale (derived from
primary ↔ reference landmark distance) so stickers stretch/shrink
naturally as the person moves closer/farther or turns.
"""

import cv2
import numpy as np
import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vision.pose_tracker import PoseTracker


# ── Colour palette (BGR for OpenCV) ──────────────────────────────────
PALETTE = [
    ((255, 255, 255), "White"),
    ((80,   80, 240), "Red"),
    ((80,  220,  80), "Green"),
    ((240,  80,  80), "Blue"),
    ((80,  240, 240), "Yellow"),
    ((220, 120, 255), "Pink"),
    ((40,  180, 255), "Orange"),
    ((200,  60, 200), "Purple"),
    ((0,     0,   0), "Black"),
]

BRUSH_SIZES = [2, 4, 7, 12, 20, 32]


# ─────────────────────────────────────────────────────────────────────
class DrawStroke:
    """A single continuous brush stroke with fixed colour and size."""

    def __init__(self, color: tuple, brush_size: int):
        self.color      = color
        self.brush_size = brush_size
        self.points: list[tuple] = []

    def add(self, x: int, y: int):
        self.points.append((x, y))

    def is_empty(self) -> bool:
        return len(self.points) < 2


# ─────────────────────────────────────────────────────────────────────
class AnchoredSticker:
    """
    A set of strokes locked to a body landmark.
    Each point is stored as a local offset from the anchor landmark
    at the moment of anchoring.  On render, translate + uniform scale
    brings points into current screen space.
    """

    def __init__(self, strokes: list, anchor_id: int, ref_id: "int|None",
                 anchor_px: tuple, ref_px: "tuple|None"):
        self.anchor_id  = anchor_id
        self.ref_id     = ref_id
        self.anchor_px  = anchor_px      # (x, y) at anchor time
        self.ref_px     = ref_px         # (x, y) at anchor time
        self.base_dist  = (math.hypot(anchor_px[0] - ref_px[0],
                                      anchor_px[1] - ref_px[1])
                           if ref_px else 1.0)

        # Convert strokes to local space (offset from anchor_px)
        self.strokes: list[DrawStroke] = []
        for s in strokes:
            ls = DrawStroke(s.color, s.brush_size)
            for (px, py) in s.points:
                ls.points.append((px - anchor_px[0], py - anchor_px[1]))
            self.strokes.append(ls)

    # ── World ↔ Local transform ───────────────────────────────────────

    def to_world(self, lx: float, ly: float,
                 cur_anchor: tuple, cur_ref: "tuple|None") -> "tuple|None":
        """Convert a local-space point to current screen coordinates."""
        if cur_anchor is None:
            return None

        # Scale from reference landmark distance
        scale = 1.0
        angle = 0.0

        if cur_ref is not None and self.ref_px is not None and self.base_dist > 1.0:
            cur_dist = math.hypot(cur_anchor[0] - cur_ref[0],
                                  cur_anchor[1] - cur_ref[1])
            scale = max(0.1, min(cur_dist / self.base_dist, 10.0))

            # Rotation from original anchor→ref angle vs current
            orig_angle = math.atan2(self.ref_px[1] - self.anchor_px[1],
                                    self.ref_px[0] - self.anchor_px[0])
            curr_angle = math.atan2(cur_ref[1] - cur_anchor[1],
                                    cur_ref[0] - cur_anchor[0])
            angle = curr_angle - orig_angle

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a

        return (int(cur_anchor[0] + rx * scale),
                int(cur_anchor[1] + ry * scale))


# ─────────────────────────────────────────────────────────────────────
class ARCanvas:
    """
    Manages free (unanchored) strokes and body-anchored stickers.

    Usage (main loop):
        canvas = ARCanvas()

        # Draw mode on/off
        canvas.toggle_draw_mode()

        # Feed finger position each frame
        if canvas.draw_mode and pointing:
            canvas.pen_at(tip_x, tip_y)
        elif canvas.draw_mode and fist:
            canvas.pen_up()

        # Render
        canvas.render(frame, pose_tracker)
    """

    def __init__(self):
        self.draw_mode = False

        self._color_idx = 0
        self._brush_idx = 1

        self._free: list[DrawStroke]      = []
        self._anchored: list[AnchoredSticker] = []
        self._current: "DrawStroke|None"  = None
        self._pen_down = False

    # ── Properties ────────────────────────────────────────────────────

    @property
    def color(self) -> tuple:
        return PALETTE[self._color_idx][0]

    @property
    def color_name(self) -> str:
        return PALETTE[self._color_idx][1]

    @property
    def brush_size(self) -> int:
        return BRUSH_SIZES[self._brush_idx]

    @property
    def has_free_strokes(self) -> bool:
        return bool(self._free) or (self._current is not None)

    # ── Controls ──────────────────────────────────────────────────────

    def toggle_draw_mode(self):
        self.draw_mode = not self.draw_mode
        if not self.draw_mode:
            self._lift_pen()

    def cycle_color(self):
        self._color_idx = (self._color_idx + 1) % len(PALETTE)

    def cycle_brush(self):
        self._brush_idx = (self._brush_idx + 1) % len(BRUSH_SIZES)

    # ── Drawing ───────────────────────────────────────────────────────

    def pen_at(self, x: int, y: int):
        """Called each frame when fingertip is detected in draw mode."""
        if not self._pen_down or self._current is None:
            self._current  = DrawStroke(self.color, self.brush_size)
            self._pen_down = True
        self._current.add(x, y)

    def pen_up(self):
        """Lift pen: finalise current stroke."""
        self._lift_pen()

    def undo(self):
        """Remove last stroke (current first, then last free)."""
        if self._current is not None:
            self._current  = None
            self._pen_down = False
        elif self._free:
            self._free.pop()

    def clear_free(self):
        self._free.clear()
        self._current  = None
        self._pen_down = False

    def clear_all(self):
        self.clear_free()
        self._anchored.clear()

    # ── Body anchoring ────────────────────────────────────────────────

    def anchor_to_body(self, pose_tracker) -> bool:
        """
        Anchor current free strokes to the nearest visible pose landmark.
        Returns True on success.
        """
        self._lift_pen()

        if not self._free:
            return False

        # Compute centroid of all stroke points
        all_pts = [pt for s in self._free for pt in s.points]
        if not all_pts:
            return False

        cx = int(sum(p[0] for p in all_pts) / len(all_pts))
        cy = int(sum(p[1] for p in all_pts) / len(all_pts))

        anchor_id = pose_tracker.closest_visible_landmark(cx, cy)
        if anchor_id is None:
            return False

        anchor_px = pose_tracker.get_point(anchor_id)
        if anchor_px is None:
            return False

        ref_id  = pose_tracker.ref_for(anchor_id)
        ref_px  = pose_tracker.get_point(ref_id) if ref_id is not None else None

        sticker = AnchoredSticker(
            self._free, anchor_id, ref_id, anchor_px, ref_px
        )
        self._anchored.append(sticker)
        self._free.clear()
        return True

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self, frame: np.ndarray,
               pose_tracker: Optional["PoseTracker"] = None) -> np.ndarray:
        """Draw all strokes and anchored stickers onto `frame`."""

        # Free strokes
        for s in self._free:
            _draw_stroke(frame, s.points, s.color, s.brush_size)

        # Active (current) stroke
        if self._current and self._current.points:
            _draw_stroke(frame, self._current.points,
                         self._current.color, self._current.brush_size)

        # Anchored stickers
        if pose_tracker is not None:
            for sticker in self._anchored:
                cur_anchor = pose_tracker.get_point(sticker.anchor_id)
                cur_ref    = (pose_tracker.get_point(sticker.ref_id)
                              if sticker.ref_id is not None else None)

                if cur_anchor is None:
                    continue

                # Compute per-sticker scale for brush width
                scale = 1.0
                if cur_ref is not None and sticker.base_dist > 1.0:
                    cd    = math.hypot(cur_anchor[0] - cur_ref[0],
                                       cur_anchor[1] - cur_ref[1])
                    scale = max(0.1, min(cd / sticker.base_dist, 10.0))

                for s in sticker.strokes:
                    world_pts = []
                    for (lx, ly) in s.points:
                        wp = sticker.to_world(lx, ly, cur_anchor, cur_ref)
                        if wp:
                            world_pts.append(wp)
                    if world_pts:
                        bs = max(1, int(s.brush_size * scale))
                        _draw_stroke(frame, world_pts, s.color, bs)

        return frame

    # ── Internal ──────────────────────────────────────────────────────

    def _lift_pen(self):
        if self._current and not self._current.is_empty():
            self._free.append(self._current)
        self._current  = None
        self._pen_down = False


# ── Rendering helper ─────────────────────────────────────────────────

def _draw_stroke(frame: np.ndarray, points: list,
                 color: tuple, brush_size: int):
    """Smooth antialiased polyline stroke."""
    if not points:
        return
    if len(points) == 1:
        cv2.circle(frame, points[0], max(1, brush_size // 2),
                   color, -1, cv2.LINE_AA)
        return

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, color, brush_size, cv2.LINE_AA)
