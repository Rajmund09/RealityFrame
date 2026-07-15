import cv2
import threading
import time
from typing import Tuple, Optional
import numpy as np

class ThreadedVideoCapture:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera index {camera_index} not found")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.ret = False
        self.frame: Optional[np.ndarray] = None
        
        self.running = True
        self.lock = threading.Lock()
        
        # Read the first frame immediately to ensure it's ready
        self.ret, self.frame = self.cap.read()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        """Continuously reads frames from the camera in a background thread."""
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            # Prevent maxing out the CPU if cap.read() fails repeatedly
            if not ret:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return self.ret, None
                
    def get(self, propId: int) -> float:
        return self.cap.get(propId)

    def release(self) -> None:
        """Stops the thread and releases the camera."""
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
