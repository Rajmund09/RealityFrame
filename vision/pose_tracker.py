"""
vision/pose_tracker.py
──────────────────────
MediaPipe Pose wrapper — runs at half resolution for speed.
Provides landmark access and body-region helpers for the AR canvas.
"""

import cv2
import mediapipe as mp
import math

# ── Landmark IDs ──────────────────────────────────────────────────────
NOSE            = 0
LEFT_EYE        = 2
RIGHT_EYE       = 5
LEFT_EAR        = 7
RIGHT_EAR       = 8
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12
LEFT_ELBOW      = 13
RIGHT_ELBOW     = 14
LEFT_WRIST      = 15
RIGHT_WRIST     = 16
LEFT_HIP        = 23
RIGHT_HIP       = 24
LEFT_KNEE       = 25
RIGHT_KNEE      = 26

# Primary → secondary (reference) landmark for scale computation
_REF_MAP = {
    NOSE:           LEFT_SHOULDER,
    LEFT_EYE:       LEFT_EAR,
    RIGHT_EYE:      RIGHT_EAR,
    LEFT_WRIST:     LEFT_ELBOW,
    RIGHT_WRIST:    RIGHT_ELBOW,
    LEFT_ELBOW:     LEFT_SHOULDER,
    RIGHT_ELBOW:    RIGHT_SHOULDER,
    LEFT_SHOULDER:  LEFT_HIP,
    RIGHT_SHOULDER: RIGHT_HIP,
    LEFT_HIP:       LEFT_KNEE,
    RIGHT_HIP:      RIGHT_KNEE,
}


class PoseTracker:
    """
    Wraps MediaPipe Pose.
    Call update() each frame; read results via get_point() / closest_landmark().
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=0,          # fastest model
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarks = None           # [(x, y, visibility), ...]
        self._skip_counter = 0

    # ── Public ────────────────────────────────────────────────────────

    def update(self, frame, every_n: int = 2):
        """
        Process `frame` and cache landmarks.
        `every_n`: only re-infer every N calls (saves compute).
        Returns current landmark list.
        """
        self._skip_counter += 1
        if self._skip_counter % every_n != 0:
            return self._landmarks

        fh, fw = frame.shape[:2]
        small  = cv2.resize(frame, (fw // 2, fh // 2))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks is not None:
            self._landmarks = [
                (int(lm.x * fw), int(lm.y * fh), lm.visibility)
                for lm in result.pose_landmarks.landmark
            ]

        return self._landmarks

    def get_point(self, lm_id: int, min_vis: float = 0.3):
        """Return (x, y) for landmark `lm_id`, or None if not visible."""
        if self._landmarks is None or lm_id >= len(self._landmarks):
            return None
        x, y, vis = self._landmarks[lm_id]
        return (x, y) if vis >= min_vis else None

    def ref_for(self, anchor_id: int):
        """Return the reference landmark ID for a given anchor."""
        return _REF_MAP.get(anchor_id)

    def closest_visible_landmark(self, cx: int, cy: int) -> "int | None":
        """ID of the visible landmark nearest to screen point (cx, cy)."""
        if self._landmarks is None:
            return None
        best_id   = None
        best_dist = float("inf")
        for i, (x, y, vis) in enumerate(self._landmarks):
            if vis < 0.3:
                continue
            d = math.hypot(x - cx, y - cy)
            if d < best_dist:
                best_dist = d
                best_id   = i
        return best_id

    def close(self):
        self.pose.close()
