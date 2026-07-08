import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )
    else:
        detector = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detector is not None:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                dictionary,
                parameters=parameters
            )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.putText(
                frame,
                "MARKER DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )
        else:
            cv2.putText(
                frame,
                "NO MARKER",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        cv2.imshow("Marker Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()