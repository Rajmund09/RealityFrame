import cv2
import numpy as np


class TargetTracker:
    def __init__(self, marker_id=0):
        self.marker_id = marker_id

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.parameters = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary,
                self.parameters
            )
        else:
            self.detector = None

        self.last_corners = None
        self.smoothing = 0.25

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.parameters
            )

        if ids is None:
            return self.last_corners

        ids = ids.flatten()

        for i, marker in enumerate(ids):
            if marker == self.marker_id:
                current = corners[i][0].astype(np.float32)

                if self.last_corners is None:
                    self.last_corners = current
                else:
                    self.last_corners = (
                        self.last_corners * (1 - self.smoothing)
                        + current * self.smoothing
                    )

                return self.last_corners

        return self.last_corners