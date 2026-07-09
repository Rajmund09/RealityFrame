class PortalDetector:
    """
    Detects the portal region from two hands using thumb+index tips.
    Applies exponential moving average smoothing to eliminate jitter.
    """

    def __init__(self):
        self._smooth = None          # (x1, y1, x2, y2) floats
        self._alpha = 0.30           # smoothing factor (lower = smoother)
        self._min_w = 130
        self._min_h = 90

    def detect(self, hands, frame_shape):
        if len(hands) < 2:
            # Slowly decay portal when hands leave
            if self._smooth is not None:
                self._smooth = None
            return None

        h, w, _ = frame_shape
        points = []

        for hand in hands:
            for lm_id in [4, 8]:        # thumb tip, index tip
                lm = hand.landmark[lm_id]
                points.append((lm.x * w, lm.y * h))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1_raw = min(xs)
        x2_raw = max(xs)
        y1_raw = min(ys)
        y2_raw = max(ys)

        if (x2_raw - x1_raw) < self._min_w or (y2_raw - y1_raw) < self._min_h:
            self._smooth = None
            return None

        raw = (x1_raw, y1_raw, x2_raw, y2_raw)

        if self._smooth is None:
            self._smooth = raw
        else:
            a = self._alpha
            self._smooth = tuple(
                self._smooth[i] * (1 - a) + raw[i] * a
                for i in range(4)
            )

        x1, y1, x2, y2 = self._smooth
        return (int(x1), int(y1), int(x2), int(y2))