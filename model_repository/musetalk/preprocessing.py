import cv2
import numpy as np
from typing import List, Tuple


def _detect_face_bbox(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """Return largest face bbox (x, y, w, h). Falls back to center crop if none found."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        h, w = gray.shape[:2]
        size = int(min(h, w) * 0.6)
        size = max(64, size)
        cx, cy = w // 2, h // 2
        x = max(0, cx - size // 2)
        y = max(0, cy - size // 2)
        size = min(size, w - x, h - y)
        return x, y, size, size
    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def get_landmark_and_bbox(
    input_img_list: List[str],
    bbox_shift: int = 0,
    max_side: int = 0,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Lightweight face bbox extractor without mmpose dependency.

    Returns:
        coord_list: list of [x1, y1, x2, y2] bbox coords
        frame_list: list of loaded BGR frames
    """
    coord_list: List[List[int]] = []
    frame_list: List[np.ndarray] = []

    for img_path in input_img_list:
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"unable to read image: {img_path}")

        if max_side and max_side > 0:
            h, w = frame.shape[:2]
            scale = max_side / float(max(h, w))
            if scale < 1.0:
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_list.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = _detect_face_bbox(gray)

        x1 = max(0, x - bbox_shift)
        y1 = max(0, y - bbox_shift)
        x2 = min(frame.shape[1], x + w + bbox_shift)
        y2 = min(frame.shape[0], y + h + bbox_shift)

        if x2 <= x1 or y2 <= y1:
            coord_list.append([0, 0, 0, 0])
        else:
            coord_list.append([int(x1), int(y1), int(x2), int(y2)])

    return coord_list, frame_list
