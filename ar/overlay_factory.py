import cv2
import numpy as np


class OverlayFactory:
    @staticmethod
    def creeper_face(size=700):
        img = np.zeros((size, size, 4), dtype=np.uint8)

        # Base Minecraft green color, RGBA
        img[:, :, 0] = 45
        img[:, :, 1] = 170
        img[:, :, 2] = 65
        img[:, :, 3] = 255

        block = size // 14

        # Pixelated texture
        for y in range(0, size, block):
            for x in range(0, size, block):
                noise = np.random.randint(-30, 30)

                color = np.array([45, 170, 65]) + noise
                color = np.clip(color, 0, 255)

                img[y:y + block, x:x + block, 0:3] = color

        black = (10, 20, 10, 255)

        def rect(x1, y1, x2, y2):
            img[y1:y2, x1:x2] = black

        # Creeper eyes
        rect(
            int(size * 0.20),
            int(size * 0.25),
            int(size * 0.42),
            int(size * 0.47)
        )

        rect(
            int(size * 0.58),
            int(size * 0.25),
            int(size * 0.80),
            int(size * 0.47)
        )

        # Nose / mouth
        rect(
            int(size * 0.42),
            int(size * 0.47),
            int(size * 0.58),
            int(size * 0.63)
        )

        rect(
            int(size * 0.30),
            int(size * 0.63),
            int(size * 0.70),
            int(size * 0.82)
        )

        rect(
            int(size * 0.22),
            int(size * 0.70),
            int(size * 0.38),
            int(size * 0.92)
        )

        rect(
            int(size * 0.62),
            int(size * 0.70),
            int(size * 0.78),
            int(size * 0.92)
        )

        # Add very subtle scan lines for AR look
        for y in range(0, size, 12):
            img[y:y + 2, :, 1] = np.clip(img[y:y + 2, :, 1] + 25, 0, 255)

        return img