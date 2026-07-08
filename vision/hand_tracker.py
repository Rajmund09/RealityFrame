import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        hands = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hands.append(hand_landmarks)

        return hands

    def draw_hands(self, frame, hands):
        for hand in hands:
            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

        return frame