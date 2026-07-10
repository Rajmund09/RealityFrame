import cv2
import numpy as np
import time
import math


# ─────────────────────────────────────────────────────────────────────
#  Premium Monochromatic Palette  (Apple / Samsung Design Language)
#  All colours in BGR for OpenCV
# ─────────────────────────────────────────────────────────────────────

# Neutrals
C_WHITE       = (240, 240, 242)   # Warm white  — primary text
C_LABEL       = (200, 200, 205)   # Secondary label
C_TERTIARY    = (130, 130, 138)   # Tertiary / captions
C_SEPARATOR   = ( 50,  52,  58)   # Hairline dividers
C_FILL        = ( 32,  33,  38)   # Panel fill
C_BG          = ( 18,  19,  22)   # Darkest surface

# Single controlled accent — barely visible, not neon
C_ACCENT      = (175, 175, 185)   # Near-white cool accent
C_ACCENT_DIM  = ( 80,  82,  92)   # Dimmed accent for borders

# Semantic
C_SUCCESS     = (120, 185, 140)   # Muted sage green
C_CAUTION     = (120, 155, 190)   # Steel blue (neutral enough)
C_ERROR       = (110, 110, 185)   # Muted mauve

MODE_COLORS = {
    "PORTAL": C_ACCENT,
    "FULL":   C_LABEL,
    "BLUR":   C_CAUTION,
}

MODE_ICONS = {
    "PORTAL": "P",
    "FULL":   "F",
    "BLUR":   "B",
}


# ─────────────────────────────────────────────────────────────────────
class Toast:
    """Temporary side-aligned notification that fades out gracefully."""

    def __init__(self, message: str, duration: float = 2.8):
        self.message  = message
        self.born     = time.time()
        self.duration = duration

    def alpha(self) -> float:
        age = time.time() - self.born
        if age < 0.18:                          # fade-in
            return age / 0.18
        if age < self.duration - 0.45:          # hold
            return 1.0
        return max(0.0, (self.duration - age) / 0.45)

    def alive(self) -> bool:
        return time.time() - self.born < self.duration


# ─────────────────────────────────────────────────────────────────────
class Renderer:

    def __init__(self):
        self._frame_idx    = 0
        self._toasts: list[Toast] = []
        self._gesture_flash = 0.0
        self._vig_cache    = None   # (w, h, vig3_ndarray) — reused every frame

    # ── Public API ───────────────────────────────────────────────────

    def push_toast(self, message: str):
        self._toasts.append(Toast(message))

    def flash_gesture(self):
        self._gesture_flash = 1.0

    def flash_mode(self):
        pass  # intentionally silent

    def tick(self):
        self._frame_idx    += 1
        self._gesture_flash = max(0.0, self._gesture_flash - 0.06)
        self._toasts        = [t for t in self._toasts if t.alive()]

    # ── Portal border ────────────────────────────────────────────────

    def draw_portal(self, frame, portal):
        if portal is None:
            return frame

        x1, y1, x2, y2 = portal

        # Very subtle pulse — just a breath, not a strobe
        pulse  = 0.65 + 0.35 * math.sin(self._frame_idx * 0.10)
        corner = 48
        thick  = 2

        line_layer = np.zeros_like(frame)

        # Ghost border
        cv2.rectangle(line_layer, (x1, y1), (x2, y2), (42, 44, 52), 1)

        # Corner L-brackets — cool grey, very clean
        base_c = (int(160 * pulse), int(162 * pulse), int(172 * pulse))
        corners = [
            ((x1, y1), (x1 + corner, y1), (x1, y1 + corner)),
            ((x2, y1), (x2 - corner, y1), (x2, y1 + corner)),
            ((x1, y2), (x1 + corner, y2), (x1, y2 - corner)),
            ((x2, y2), (x2 - corner, y2), (x2, y2 - corner)),
        ]
        for main, p1, p2 in corners:
            cv2.line(line_layer, main, p1, base_c, thick, cv2.LINE_AA)
            cv2.line(line_layer, main, p2, base_c, thick, cv2.LINE_AA)

        # Single-pass soft glow — very restrained
        glow = cv2.GaussianBlur(line_layer, (9, 9), 0)
        frame = cv2.addWeighted(frame, 1.0, glow,       0.25 * pulse, 0)
        frame = cv2.addWeighted(frame, 1.0, line_layer, 0.90,         0)

        return frame

    # ── Main HUD ─────────────────────────────────────────────────────

    def draw_hud(self, frame, mode, portal, fps=0,
                 ar_mode=False, focus_mode=False, focus_box=None,
                 selecting=False, gesture_name=None,
                 zoom_level=1.0, vcam_active=False, custom_bg_name=None,
                 draw_mode=False, canvas=None):

        h, w = frame.shape[:2]

        # ── Vignette for depth (very subtle) ──────────────────────────
        frame = self._draw_vignette(frame, w, h)

        # ── 1-px hairline top rule ────────────────────────────────────
        frame = self._draw_top_rule(frame, mode, w)

        # ── Status panel (top-left) ───────────────────────────────────
        frame = self._draw_status_panel(frame, mode, fps, ar_mode,
                                        focus_mode, focus_box,
                                        zoom_level, vcam_active,
                                        custom_bg_name)

        # ── Gesture badge (top-right, fades naturally) ────────────────
        if self._gesture_flash > 0.05 and gesture_name:
            frame = self._draw_gesture_badge(frame, gesture_name, w)

        # ── Focus selection hint — LEFT side, not centre ─────────────
        if selecting:
            frame = self._draw_select_hint(frame, w, h)

        # ── Bottom control bar ────────────────────────────────────────
        frame = self._draw_control_bar(frame, w, h, mode)

        # ── Toasts — right side ───────────────────────────────────────
        frame = self._draw_toasts(frame, w, h)

        return frame

    # ── Focus selection dots + lines ────────────────────────────────

    def draw_focus_points(self, frame, points):
        ring_c = C_ACCENT
        dot_c  = C_WHITE
        line_c = C_ACCENT_DIM

        for i, pt in enumerate(points):
            cv2.circle(frame, pt, 12, ring_c, 1, cv2.LINE_AA)
            cv2.circle(frame, pt,  4, dot_c,  -1, cv2.LINE_AA)
            cv2.putText(frame, str(i + 1),
                        (pt[0] + 16, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_TERTIARY, 1, cv2.LINE_AA)

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], line_c, 1, cv2.LINE_AA)

        if len(points) == 4:
            cv2.line(frame, points[3], points[0], line_c, 1, cv2.LINE_AA)

        return frame

    # ── Hand skeleton ─────────────────────────────────────────────────

    def draw_hands(self, frame, hands, mp_hands, mp_draw):
        spec_lm   = mp_draw.DrawingSpec(color=C_ACCENT,    thickness=1, circle_radius=3)
        spec_conn = mp_draw.DrawingSpec(color=C_SEPARATOR,  thickness=1, circle_radius=1)
        for hand in hands:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                spec_lm, spec_conn
            )
        return frame

    # ─────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _glass_rect(self, frame, x1, y1, x2, y2,
                    alpha: float = 0.72,
                    color=C_FILL,
                    radius: int = 10,
                    border_color=C_SEPARATOR):
        """
        Frosted-glass panel — dark, clean, no colour bleed.
        Uses a true rounded-rect fill then a hairline border.
        """
        overlay = frame.copy()

        # Body fill with rounded corners
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius),
                       (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius),
                       (x2 - radius, y2 - radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)

        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # Hairline border
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

        return frame

    # ── Status panel ─────────────────────────────────────────────────

    def _draw_status_panel(self, frame, mode, fps, ar_mode, focus_mode,
                            focus_box, zoom_level=1.0,
                            vcam_active=False, custom_bg_name=None):
        pw, ph = 230, 130
        px, py = 14, 14
        frame = self._glass_rect(frame, px, py, px + pw, py + ph)

        accent = MODE_COLORS.get(mode, C_ACCENT)

        # Subtle left accent rule
        cv2.line(frame,
                 (px + 1, py + 10),
                 (px + 1, py + ph - 10),
                 accent, 2, cv2.LINE_AA)

        # Mode label
        icon  = MODE_ICONS.get(mode, "◈")
        label = f"{icon}  {mode}"
        cv2.putText(frame, label,
                    (px + 16, py + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)

        # Hairline divider
        cv2.line(frame,
                 (px + 14, py + 46),
                 (px + pw - 14, py + 46),
                 C_SEPARATOR, 1)

        # FPS
        fps_label = f"FPS   {fps:3d}"
        cv2.putText(frame, fps_label,
                    (px + 16, py + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TERTIARY, 1, cv2.LINE_AA)

        # Zoom level (shown only when != 1.0)
        if zoom_level > 1.05:
            z_str = f"ZOOM  {zoom_level:.1f}×"
            cv2.putText(frame, z_str,
                        (px + 16, py + 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TERTIARY, 1, cv2.LINE_AA)

        # Status pills row
        pill_x  = px + 16
        pill_y1 = py + 98
        pill_y2 = py + 118

        pills = []
        if ar_mode:
            pills.append(("AR",      C_CAUTION))
        if focus_mode and focus_box:
            pills.append(("FOCUS",   C_SUCCESS))
        if vcam_active:
            pills.append(("VCAM",    C_ACCENT))
        if custom_bg_name:
            pills.append(("CUSTOM",  C_LABEL))

        for label, c in pills:
            tw  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
            pad = 6
            cv2.rectangle(frame,
                          (pill_x, pill_y1),
                          (pill_x + tw + pad * 2, pill_y2),
                          c, 1, cv2.LINE_AA)
            cv2.putText(frame, label,
                        (pill_x + pad, pill_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1, cv2.LINE_AA)
            pill_x += tw + pad * 2 + 8

        return frame

    # ── Gesture badge ─────────────────────────────────────────────────

    def _draw_gesture_badge(self, frame, name: str, w: int):
        label = name.upper()
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        px2 = w - 14
        px1 = px2 - tw - 28
        py1, py2 = 14, 46

        frame = self._glass_rect(frame, px1, py1, px2, py2)
        a     = self._gesture_flash
        c     = tuple(int(v * min(a, 1.0)) for v in C_LABEL)
        cv2.putText(frame, label,
                    (px1 + 14, py2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1, cv2.LINE_AA)
        return frame

    # ── Control bar ───────────────────────────────────────────────────

    def _draw_control_bar(self, frame, w: int, h: int, mode: str,
                           draw_mode: bool = False):
        """Slim bottom bar — key badges adapt to current mode."""
        bh  = 34
        by1 = h - bh - 8
        by2 = h - 8
        frame = self._glass_rect(frame, 10, by1, w - 10, by2, alpha=0.65)

        if draw_mode:
            binds = [
                ("D",  "Exit Draw"),
                ("C",  "Colour"),
                ("W",  "Brush Size"),
                ("N",  "Stick to Body"),
                ("U",  "Undo"),
                ("X",  "Clear"),
            ]
        else:
            binds = [
                ("Q",  "Quit"),
                ("B",  "Recapture"),
                ("R",  "Focus"),
                ("A",  "AR"),
                ("C",  "VCam"),
                ("I",  "Custom BG"),
                ("D",  "Draw Mode"),
                ("Z",  "Zoom Reset"),
            ]

        x = 20
        for key, desc in binds:
            ksize = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            kw    = ksize[0]
            bx1   = x
            bx2   = x + kw + 10
            by_k1 = by1 + 6
            by_k2 = by2 - 6

            cv2.rectangle(frame, (bx1, by_k1), (bx2, by_k2),
                          C_SEPARATOR, -1)
            cv2.putText(frame, key,
                        (bx1 + 5, by2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_LABEL, 1, cv2.LINE_AA)
            x = bx2 + 8

            dw = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
            cv2.putText(frame, desc,
                        (x, by2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TERTIARY, 1, cv2.LINE_AA)
            x += dw + 18

            if x > w - 40:
                break

        return frame

    # ── Draw-mode canvas palette UI ─────────────────────────────────

    def _draw_canvas_ui(self, frame, w: int, h: int, canvas):
        """
        Right-side palette panel shown when draw mode is active.
        Shows: DRAW badge, current colour swatch, brush preview,
        colour palette chips.
        """
        from core.ar_canvas import PALETTE, BRUSH_SIZES

        panel_w = 72
        pad     = 10
        px1     = w - panel_w - pad
        py1     = 60

        swatch_sz  = 24
        chip_sz    = 16
        chip_gap   = 4
        palette_h  = len(PALETTE) * (chip_sz + chip_gap)
        brush_area = 50
        total_h    = 30 + 10 + swatch_sz + 10 + brush_area + 10 + palette_h + 10
        px2        = px1 + panel_w
        py2        = py1 + total_h

        frame = self._glass_rect(frame, px1, py1, px2, py2, alpha=0.80)

        cy = py1 + 10

        # ── DRAW badge ───────────────────────────────────────────
        cv2.putText(frame, "DRAW",
                    (px1 + 10, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 1, cv2.LINE_AA)
        cy += 28
        cv2.line(frame, (px1 + 8, cy), (px2 - 8, cy), C_SEPARATOR, 1)
        cy += 8

        # ── Current colour swatch ────────────────────────────────
        sx   = px1 + (panel_w - swatch_sz) // 2
        cv2.rectangle(frame, (sx, cy), (sx + swatch_sz, cy + swatch_sz),
                      canvas.color, -1)
        cv2.rectangle(frame, (sx, cy), (sx + swatch_sz, cy + swatch_sz),
                      C_ACCENT, 1)
        cy += swatch_sz + 8

        # ── Brush size preview ──────────────────────────────────
        bcx = px1 + panel_w // 2
        bcy = cy + brush_area // 2
        br  = max(1, canvas.brush_size // 2)
        cv2.circle(frame, (bcx, bcy), br, canvas.color, -1, cv2.LINE_AA)
        cy += brush_area + 6
        cv2.line(frame, (px1 + 8, cy), (px2 - 8, cy), C_SEPARATOR, 1)
        cy += 6

        # ── Colour palette chips ────────────────────────────────
        cpx = px1 + (panel_w - chip_sz) // 2
        for i, (c, _) in enumerate(PALETTE):
            is_selected = (i == canvas._color_idx)
            cv2.rectangle(frame,
                          (cpx, cy),
                          (cpx + chip_sz, cy + chip_sz),
                          c, -1)
            if is_selected:
                cv2.rectangle(frame,
                              (cpx - 2, cy - 2),
                              (cpx + chip_sz + 2, cy + chip_sz + 2),
                              C_WHITE, 1)
            cy += chip_sz + chip_gap

        return frame

    # ── Select hint — LEFT side, not centre ──────────────────────────

    def _draw_select_hint(self, frame, w: int, h: int):
        """
        Apple Maps-style instruction strip — left-aligned, not centred.
        No bright background, no neon text.
        """
        msg  = "Click 4 corners to define focus region"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.52
        tw, th = cv2.getTextSize(msg, font, fs, 1)[0]

        margin = 14
        pad_x, pad_y = 14, 10

        # Anchor to LEFT side, vertically centred-ish
        rx1 = margin
        ry1 = h // 2 - th // 2 - pad_y - 6
        rx2 = rx1 + tw + pad_x * 2
        ry2 = ry1 + th + pad_y * 2 + 6

        frame = self._glass_rect(frame, rx1, ry1, rx2, ry2, alpha=0.78)

        # Left accent rule inside panel
        cv2.line(frame,
                 (rx1 + 2, ry1 + 6),
                 (rx1 + 2, ry2 - 6),
                 C_ACCENT_DIM, 2, cv2.LINE_AA)

        cv2.putText(frame, msg,
                    (rx1 + pad_x + 6, ry1 + th + pad_y),
                    font, fs, C_LABEL, 1, cv2.LINE_AA)

        return frame

    # ── Toasts — right side ───────────────────────────────────────────

    def _draw_toasts(self, frame, w: int, h: int):
        """
        Notification toasts anchored to the bottom-RIGHT edge.
        Clean white text on frosted dark panel, no colour highlight.
        """
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.50
        pad_x, pad_y = 16, 10
        margin       = 14

        y_bottom = h - 54  # above the control bar

        for toast in reversed(self._toasts):
            a   = toast.alpha()
            msg = toast.message

            tw, th = cv2.getTextSize(msg, font, fs, 1)[0]

            rx2 = w - margin
            rx1 = rx2 - tw - pad_x * 2
            ry2 = y_bottom
            ry1 = ry2 - th - pad_y * 2

            # Glass panel with controlled alpha
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), C_FILL, -1)
            cv2.line(overlay, (rx1, ry1), (rx2, ry1), C_SEPARATOR, 1)
            frame = cv2.addWeighted(frame, 1 - a * 0.78, overlay, a * 0.78, 0)

            # Text — simple white, fades with alpha
            text_c = tuple(int(v * a) for v in C_LABEL)
            cv2.putText(frame, msg,
                        (rx1 + pad_x, ry2 - pad_y - 1),
                        font, fs, text_c, 1, cv2.LINE_AA)

            y_bottom = ry1 - 8

        return frame

    # ── Vignette ─────────────────────────────────────────────────────

    def _draw_vignette(self, frame, w: int, h: int):
        """
        Precomputed dark-edge vignette — computed once per resolution,
        then reused every frame via a cached float32 multiplier.
        """
        if self._vig_cache is None or self._vig_cache[0] != (w, h):
            cx, cy = w // 2, h // 2
            Y, X   = np.ogrid[:h, :w]
            dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
            vig    = np.clip(1.0 - dist * 0.42, 0.72, 1.0).astype(np.float32)
            vig3   = cv2.merge([vig, vig, vig])
            self._vig_cache = ((w, h), vig3)

        _, vig3 = self._vig_cache
        return (frame.astype(np.float32) * vig3).astype(np.uint8)

    # ── Top hairline rule ─────────────────────────────────────────────

    def _draw_top_rule(self, frame, mode: str, w: int):
        """
        Single 1-px hairline at top — barely visible.
        No glow, no pulsing neon bar.
        """
        pulse = 0.55 + 0.45 * math.sin(self._frame_idx * 0.06)
        c     = tuple(int(v * pulse) for v in C_ACCENT_DIM)
        cv2.line(frame, (0, 0), (w, 0), c, 1)
        return frame