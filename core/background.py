import cv2
import numpy as np
import time


# ── Palette (matches renderer.py) ────────────────────────────────────
C_FILL      = ( 32,  33,  38)
C_SEPARATOR = ( 50,  52,  58)
C_WHITE     = (240, 240, 242)
C_LABEL     = (200, 200, 205)
C_TERTIARY  = (130, 130, 138)
C_ACCENT    = (175, 175, 185)


class BackgroundModel:
    def __init__(self):
        self.background = None

    def capture(self, cap, frames_count=180):
        """
        Capture background with a 3-second countdown so the user can
        step out of frame before sampling begins.
        """
        self._run_countdown(cap)
        self._sample_frames(cap, frames_count)

    # ------------------------------------------------------------------
    def _run_countdown(self, cap):
        """Clean, Apple-style 3-second countdown overlay."""
        start    = time.time()
        duration = 3.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            elapsed = time.time() - start
            remaining = max(0.0, duration - elapsed)

            if remaining <= 0:
                break

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            # Subtle dark scrim — not a full blackout
            scrim = frame.copy()
            cv2.rectangle(scrim, (0, 0), (w, h), (10, 11, 14), -1)
            frame = cv2.addWeighted(frame, 0.45, scrim, 0.55, 0)

            # Progress ring — muted grey track, cool-grey fill
            angle = int((1.0 - (remaining % 1.0)) * 360)
            radius = 72
            ry = cy - 20

            # Track (dim)
            cv2.ellipse(frame, (cx, ry), (radius, radius),
                        -90, 0, 360, C_SEPARATOR, 6, cv2.LINE_AA)
            # Sweep (accent — cool grey, NOT neon)
            cv2.ellipse(frame, (cx, ry), (radius, radius),
                        -90, 0, angle, C_ACCENT, 6, cv2.LINE_AA)

            # Large countdown digit
            num = str(int(remaining) + 1)
            ts  = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 5)[0]
            cv2.putText(frame, num,
                        (cx - ts[0] // 2, ry + ts[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.2, C_WHITE, 5, cv2.LINE_AA)

            # Instruction — small, calm, left-of-centre strip
            msg = "Step out of frame — capturing background"
            ms  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
            mx  = cx - ms[0] // 2
            my  = ry + radius + 36
            cv2.putText(frame, msg,
                        (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TERTIARY, 1, cv2.LINE_AA)

            cv2.imshow("RealityFrame", frame)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    def _sample_frames(self, cap, frames_count: int):
        """Sample frames — clean progress strip at the bottom, no neon."""
        frames = []

        for i in range(frames_count):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frames.append(frame.copy())

            h, w = frame.shape[:2]
            progress = (i + 1) / frames_count

            # Subtle dim
            dim = (frame * 0.65).astype(np.uint8)

            # ── Progress strip ──────────────────────────────────────
            bar_h  = 3                   # hairline
            bar_x1 = 60
            bar_x2 = w - 60
            bar_y  = h - 48

            # Track
            cv2.rectangle(dim,
                          (bar_x1, bar_y),
                          (bar_x2, bar_y + bar_h),
                          C_SEPARATOR, -1)
            # Fill — cool grey, NOT teal
            fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * progress)
            cv2.rectangle(dim,
                          (bar_x1, bar_y),
                          (fill_x2, bar_y + bar_h),
                          C_ACCENT, -1)

            # Percentage — right-aligned, tertiary
            pct_label = f"{int(progress * 100)}%"
            ps = cv2.getTextSize(pct_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            cv2.putText(dim, pct_label,
                        (bar_x2 - ps[0], bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TERTIARY, 1, cv2.LINE_AA)

            # Label — left-aligned
            label = "Calibrating background"
            cv2.putText(dim, label,
                        (bar_x1, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TERTIARY, 1, cv2.LINE_AA)

            cv2.imshow("RealityFrame", dim)
            cv2.waitKey(1)

        if frames:
            self.background = np.median(frames, axis=0).astype(np.uint8)

    def get(self):
        return self.background