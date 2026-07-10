import cv2
import numpy as np


class ARRenderer:
    def __init__(self, overlay_rgba):
        self.overlay = overlay_rgba

    def draw_target_overlay(self, frame, corners):
        if corners is None:
            return frame

        output = frame.copy()

        h, w = self.overlay.shape[:2]

        src = np.array(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ],
            dtype=np.float32
        )

        dst = corners.astype(np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
            self.overlay,
            matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        rgb = warped[:, :, :3]
        alpha = warped[:, :, 3].astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

        alpha_3d = cv2.merge([alpha, alpha, alpha])

        output = (
            rgb.astype(np.float32) * alpha_3d
            + output.astype(np.float32) * (1 - alpha_3d)
        ).astype(np.uint8)

        return output

    def draw_tracking_frame(self, frame, corners):
        if corners is None:
            return frame

        pts = corners.astype(int)

        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 120), 2)

        for p in pts:
            cv2.circle(frame, tuple(p), 5, (0, 255, 120), -1)

        return frame