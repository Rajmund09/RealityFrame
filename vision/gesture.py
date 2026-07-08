import math
import time


class GestureController:
    """
    Detects hand gestures from MediaPipe landmarks.

    Gestures supported:
      - pinch (1 hand)  : continuous thumb↔index distance → zoom level
      - pinch (2 hands) : both hands pinch simultaneously → cycle mode
      - open_palm       : all 5 fingers extended → pause effect
      - peace           : index + middle extended → start focus selection
    """

    def __init__(self):
        self._cooldowns = {}          # gesture_name -> last_trigger_time
        self._cooldown_time = 1.0     # seconds between same-gesture fires

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pinch_triggered(self, hands):
        """Single-hand pinch trigger (kept for backward compat)."""
        return self._check_any(hands, self._is_pinch, "pinch")

    def two_hand_pinch_triggered(self, hands) -> bool:
        """
        Fires only when BOTH hands are present and BOTH are pinching.
        Used for mode cycling so it never conflicts with 1-hand zoom.
        """
        if len(hands) < 2:
            return False
        now  = time.time()
        last = self._cooldowns.get("two_pinch", 0)
        if now - last < self._cooldown_time:
            return False
        if self._is_pinch(hands[0]) and self._is_pinch(hands[1]):
            self._cooldowns["two_pinch"] = now
            return True
        return False

    def peace_triggered(self, hands):
        return self._check_any(hands, self._is_peace, "peace")

    # Live state queries (no cooldown — used for portal detection etc.)
    def any_pinch(self, hands):
        return any(self._is_pinch(h) for h in hands)

    @staticmethod
    def pinch_distance(hand) -> float:
        """
        Return normalised thumb↔index distance for a single hand.
        ~0.0 = fully closed pinch,  ~1.0 = fingers fully spread.
        Used by ZoomController for continuous zoom mapping.
        """
        thumb    = hand.landmark[4]
        index    = hand.landmark[8]
        wrist    = hand.landmark[0]
        mid_mcp  = hand.landmark[9]
        pinch    = math.hypot(thumb.x - index.x, thumb.y - index.y)
        size     = math.hypot(wrist.x - mid_mcp.x, wrist.y - mid_mcp.y)
        if size < 1e-6:
            return 0.5
        return min(pinch / (size * 0.65), 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_any(self, hands, detector_fn, name):
        now = time.time()
        last = self._cooldowns.get(name, 0)
        if now - last < self._cooldown_time:
            return False
        for hand in hands:
            if detector_fn(hand):
                self._cooldowns[name] = now
                return True
        return False

    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    @staticmethod
    def _finger_extended(hand, tip_id, pip_id):
        """Returns True if finger tip is above (lower y) its PIP joint."""
        return hand.landmark[tip_id].y < hand.landmark[pip_id].y

    def _is_pinch(self, hand):
        thumb = hand.landmark[4]
        index = hand.landmark[8]
        wrist = hand.landmark[0]
        mid_mcp = hand.landmark[9]
        pinch_dist = self._dist(thumb, index)
        hand_size = self._dist(wrist, mid_mcp)
        return pinch_dist < hand_size * 0.35

    def _is_open_palm(self, hand):
        """All 4 fingers + thumb extended."""
        tips =  [4,  8, 12, 16, 20]
        pips =  [3,  6, 10, 14, 18]
        # Thumb uses x-axis comparison instead
        thumb_open = (hand.landmark[4].x < hand.landmark[3].x or
                      hand.landmark[4].x > hand.landmark[3].x)  # always ok
        # Simpler: just check thumb tip above base
        fingers_extended = all(
            hand.landmark[tips[i]].y < hand.landmark[pips[i]].y
            for i in range(1, 5)
        )
        # Thumb: tip far from palm base
        thumb_ext = self._dist(hand.landmark[4], hand.landmark[0]) > \
                    self._dist(hand.landmark[9], hand.landmark[0]) * 0.7
        return fingers_extended and thumb_ext

    def _is_peace(self, hand):
        """Index + middle up, rest curled."""
        index_up  = self._finger_extended(hand, 8,  6)
        middle_up = self._finger_extended(hand, 12, 10)
        ring_curled  = not self._finger_extended(hand, 16, 14)
        pinky_curled = not self._finger_extended(hand, 20, 18)
        return index_up and middle_up and ring_curled and pinky_curled

    def _is_pointing(self, hand):
        """Index finger only extended — draw gesture."""
        return (self._finger_extended(hand, 8,  6) and
                not self._finger_extended(hand, 12, 10) and
                not self._finger_extended(hand, 16, 14) and
                not self._finger_extended(hand, 20, 18))

    def _is_fist(self, hand):
        """All fingers curled — pen-up / stop drawing."""
        return (not self._finger_extended(hand, 8,  6) and
                not self._finger_extended(hand, 12, 10) and
                not self._finger_extended(hand, 16, 14) and
                not self._finger_extended(hand, 20, 18))

    # ── Draw-mode live queries (no cooldown) ──────────────────────────

    def pointing_hand(self, hands):
        """Return the first pointing hand landmark object, or None."""
        for hand in hands:
            if self._is_pointing(hand):
                return hand
        return None

    def fist_any(self, hands) -> bool:
        """True if any hand is a fist."""
        return any(self._is_fist(h) for h in hands)