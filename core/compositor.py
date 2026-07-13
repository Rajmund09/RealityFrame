import cv2
import numpy as np
from typing import Tuple, Optional, Any

_bg_light_cache: Optional[np.ndarray] = None
_bg_light_tick: int = 0
_BG_LIGHT_INTERVAL: int = 6

def make_person_mask(segmenter: Any, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (hard uint8 0/255, soft float32 0-1) person mask.
    Runs at HALF resolution internally for ~4x speedup, then upscales.
    """
    fh, fw = frame.shape[:2]
    small = cv2.resize(frame, (fw // 2, fh // 2))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)

    raw = (result.segmentation_mask > 0.55).astype(np.uint8)

    k7 = np.ones((7, 7), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k7)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k3)

    hard_full = cv2.resize(
        (cleaned * 255).astype(np.uint8), (fw, fh),
        interpolation=cv2.INTER_NEAREST
    )
    soft_small = cv2.GaussianBlur(cleaned.astype(np.float32), (21, 21), 0)
    soft_full = cv2.resize(soft_small, (fw, fh), interpolation=cv2.INTER_LINEAR)
    soft_full = np.clip(soft_full * 1.5, 0.0, 1.0)

    return hard_full, soft_full

def invalidate_bg_cache() -> None:
    global _bg_light_cache, _bg_light_tick
    _bg_light_cache = None
    _bg_light_tick = 0

def match_background_light(background: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Per-channel brightness correction, cached every N frames."""
    global _bg_light_cache, _bg_light_tick
    _bg_light_tick += 1
    if _bg_light_cache is None or _bg_light_tick % _BG_LIGHT_INTERVAL == 0:
        bg = background.astype(np.float32)
        live = frame.astype(np.float32)
        # Vectorized brightness adjustment per channel
        mean_diff = live.mean(axis=(0, 1)) - bg.mean(axis=(0, 1))
        bg += mean_diff
        _bg_light_cache = np.clip(bg, 0, 255).astype(np.uint8)
    return _bg_light_cache

def apply_virtual_background(frame: np.ndarray, bg_frame: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """
    Zoom / Teams -style virtual background:
    Person stays fully visible; everything outside is replaced by bg_frame.
    soft_mask: 1.0 = person, 0.0 = background.
    """
    alpha = soft_mask[:, :, np.newaxis]
    output = (frame.astype(np.float32) * alpha + bg_frame.astype(np.float32) * (1.0 - alpha))
    return np.clip(output, 0, 255).astype(np.uint8)

def apply_full_invisibility(frame: np.ndarray, background: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """Person becomes invisible - background shows everywhere."""
    corrected_bg = match_background_light(background, frame)
    alpha = soft_mask[:, :, np.newaxis]
    # Person area -> bg (person invisible), bg area -> live frame
    output = (corrected_bg.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha))
    return np.clip(output, 0, 255).astype(np.uint8)

def apply_portal_invisibility(frame: np.ndarray, background: np.ndarray, portal: Optional[Tuple[int, int, int, int]], soft_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Portal rectangle shows background behind the person."""
    if portal is None:
        return frame.copy()

    corrected_bg = match_background_light(background, frame)
    output = frame.copy()

    x1, y1, x2, y2 = portal
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

    if soft_mask is not None:
        region_alpha = soft_mask[y1:y2, x1:x2, np.newaxis]
        region_bg = corrected_bg[y1:y2, x1:x2].astype(np.float32)
        region_frame = output[y1:y2, x1:x2].astype(np.float32)
        blended = (region_bg * region_alpha + region_frame * (1.0 - region_alpha))
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        output[y1:y2, x1:x2] = corrected_bg[y1:y2, x1:x2]

    return output

def apply_background_blur(frame: np.ndarray, soft_mask: np.ndarray, blur_strength: int = 31) -> np.ndarray:
    """Teams-style background blur - person sharp, background blurred."""
    k = max(1, blur_strength) | 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    alpha = soft_mask[:, :, np.newaxis]
    output = (frame.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha))
    return np.clip(output, 0, 255).astype(np.uint8)

def apply_focus_window(frame: np.ndarray, background: np.ndarray, box: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Outside the focus box -> replaced with background."""
    if box is None:
        return frame

    corrected_bg = match_background_light(background, frame)
    x1, y1, x2, y2 = box

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    mask3d = mask[:, :, np.newaxis].astype(np.float32) / 255.0

    output = (frame.astype(np.float32) * mask3d + corrected_bg.astype(np.float32) * (1.0 - mask3d))
    return output.astype(np.uint8)
