import cv2
import numpy as np
import os


def main():
    os.makedirs("assets", exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    marker_id = 0
    marker_size = 700
    margin = 120

    marker = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        marker_size
    )

    sheet_size = marker_size + margin * 2

    # White sheet around marker is important for detection
    sheet = np.ones((sheet_size, sheet_size), dtype=np.uint8) * 255

    sheet[
        margin:margin + marker_size,
        margin:margin + marker_size
    ] = marker

    path = "assets/marker_0.png"
    cv2.imwrite(path, sheet)

    print(f"Marker saved to: {path}")
    print("Open this image full screen on phone or print it.")


if __name__ == "__main__":
    main()