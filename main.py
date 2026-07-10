import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog

from core.background import BackgroundModel
from core.custom_background import CustomBackground
from core.virtual_cam import VirtualCamOutput
from core.zoom import ZoomController
from core.ar_canvas import ARCanvas
from vision.hand_tracker import HandTracker
from vision.portal_detector import PortalDetector
from vision.gesture import GestureController
from vision.pose_tracker import PoseTracker
from graphics.renderer import Renderer

from ar.target_tracker import TargetTracker
from ar.overlay_factory import OverlayFactory
from ar.ar_renderer import ARRenderer


# ─────────────────────────────────────────────────────────────────────
focus_points = []
MODES = ["PORTAL", "FULL", "BLUR"]   # mode cycle


def mouse_callback(event, x, y, flags, param):
    global focus_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(focus_points) < 4:
            focus_points.append((x, y))


# ─────────────────────────────────────────────────────────────────────
#  Segmentation & masking
# ─────────────────────────────────────────────────────────────────────

def make_person_mask(segmenter, frame):
    """
    Returns (hard uint8 0/255, soft float32 0-1) person mask.
    Runs at HALF resolution internally for ~4× speedup, then upscales.
    """
    fh, fw = frame.shape[:2]
    small  = cv2.resize(frame, (fw // 2, fh // 2))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)

    raw = (result.segmentation_mask > 0.55).astype(np.uint8)

    k7 = np.ones((7, 7), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k7)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k3)

    hard_full = cv2.resize(
        (cleaned * 255).astype(np.uint8), (fw, fh),
        interpolation=cv2.INTER_NEAREST
    )
    soft_small = cv2.GaussianBlur(cleaned.astype(np.float32), (21, 21), 0)
    soft_full  = cv2.resize(soft_small, (fw, fh),
                            interpolation=cv2.INTER_LINEAR)
    soft_full  = np.clip(soft_full * 1.5, 0.0, 1.0)

    return hard_full, soft_full


# ── Light-match cache ─────────────────────────────────────────────────
_bg_light_cache = None
_bg_light_tick  = 0
_BG_LIGHT_INTERVAL = 6


def invalidate_bg_cache():
    global _bg_light_cache, _bg_light_tick
    _bg_light_cache = None
    _bg_light_tick  = 0


def match_background_light(background, frame):
    """Per-channel brightness correction, cached every N frames."""
    global _bg_light_cache, _bg_light_tick
    _bg_light_tick += 1
    if _bg_light_cache is None or _bg_light_tick % _BG_LIGHT_INTERVAL == 0:
        bg   = background.astype(np.float32)
        live = frame.astype(np.float32)
        for c in range(3):
            bg[:, :, c] += (live[:, :, c].mean() - bg[:, :, c].mean())
        _bg_light_cache = np.clip(bg, 0, 255).astype(np.uint8)
    return _bg_light_cache


# ─────────────────────────────────────────────────────────────────────
#  Compositing functions
# ─────────────────────────────────────────────────────────────────────

def apply_virtual_background(frame, bg_frame, soft_mask):
    """
    Zoom / Teams -style virtual background:
    Person stays fully visible; everything outside is replaced by bg_frame.
    soft_mask: 1.0 = person, 0.0 = background.
    """
    alpha  = soft_mask[:, :, np.newaxis]
    output = (frame.astype(np.float32) * alpha
              + bg_frame.astype(np.float32) * (1.0 - alpha))
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_full_invisibility(frame, background, soft_mask):
    """Person becomes invisible — background shows everywhere."""
    corrected_bg = match_background_light(background, frame)
    alpha  = soft_mask[:, :, np.newaxis]
    # Person area → bg (person invisible), bg area → live frame
    output = (corrected_bg.astype(np.float32) * alpha
              + frame.astype(np.float32) * (1.0 - alpha))
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_portal_invisibility(frame, background, portal,
                               soft_mask=None):
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
        region_bg    = corrected_bg[y1:y2, x1:x2].astype(np.float32)
        region_frame = output[y1:y2, x1:x2].astype(np.float32)
        blended = (region_bg * region_alpha
                   + region_frame * (1.0 - region_alpha))
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        output[y1:y2, x1:x2] = corrected_bg[y1:y2, x1:x2]

    return output


def apply_background_blur(frame, soft_mask, blur_strength: int = 31):
    """Teams-style background blur — person sharp, background blurred."""
    k       = max(1, blur_strength) | 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    alpha   = soft_mask[:, :, np.newaxis]
    output  = (frame.astype(np.float32) * alpha
               + blurred.astype(np.float32) * (1.0 - alpha))
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_focus_window(frame, background, box):
    """Outside the focus box → replaced with background."""
    if box is None:
        return frame

    corrected_bg = match_background_light(background, frame)
    x1, y1, x2, y2 = box

    mask   = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask   = cv2.GaussianBlur(mask, (41, 41), 0)
    mask3d = mask[:, :, np.newaxis].astype(np.float32) / 255.0

    output = (frame.astype(np.float32) * mask3d
              + corrected_bg.astype(np.float32) * (1.0 - mask3d))
    return output.astype(np.uint8)


def flip_corners_for_mirror(corners, frame_width):
    if corners is None:
        return None
    flipped = corners.copy()
    flipped[:, 0] = frame_width - flipped[:, 0]
    flipped = np.array([flipped[1], flipped[0], flipped[3], flipped[2]],
                       dtype=np.float32)
    return flipped


# ─────────────────────────────────────────────────────────────────────
#  Custom background — native file-picker dialog
# ─────────────────────────────────────────────────────────────────────

_bg_prompt_active = False


def _open_bg_picker(custom_bg: CustomBackground, renderer: Renderer):
    """Native file-picker dialog (runs in daemon thread)."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = [
        ("All supported",
         "*.jpg *.jpeg *.png *.bmp *.webp *.tiff "
         "*.mp4 *.avi *.mov *.mkv *.webm"),
        ("Images",  "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
        ("Videos",  "*.mp4 *.avi *.mov *.mkv *.webm"),
        ("All files", "*.*"),
    ]

    path = filedialog.askopenfilename(
        title     = "Select Background Image or Video",
        filetypes = filetypes,
    )
    root.destroy()

    if path:
        ok = custom_bg.set_source(path)
        invalidate_bg_cache()          # ← clear stale light-match cache
        renderer.push_toast(
            f"BG: {custom_bg.source_name}" if ok else "Could not load file"
        )
    else:
        renderer.push_toast("No file selected")


# ─────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────

def main():
    global focus_points, _bg_prompt_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("RealityFrame")
    cv2.setMouseCallback("RealityFrame", mouse_callback)

    # ── Background capture ────────────────────────────────────────────
    background_model = BackgroundModel()
    background_model.capture(cap, frames_count=180)
    background = background_model.get()

    # ── Initialise components ─────────────────────────────────────────
    hand_tracker    = HandTracker()
    portal_detector = PortalDetector()
    gesture         = GestureController()
    renderer        = Renderer()
    zoom_ctrl       = ZoomController()
    custom_bg       = CustomBackground()
    vcam            = VirtualCamOutput(actual_w, actual_h, fps=30)
    canvas          = ARCanvas()
    pose_tracker    = PoseTracker()

    target_tracker  = TargetTracker(marker_id=0)
    ar_overlay      = OverlayFactory.creeper_face(size=700)
    ar_renderer_obj = ARRenderer(ar_overlay)

    mp_hands_mod = mp.solutions.hands
    mp_draw      = mp.solutions.drawing_utils

    # ── State ─────────────────────────────────────────────────────────
    mode_idx        = 0
    invisible_mode  = MODES[mode_idx]

    blur_strength   = 31
    zoom_enabled    = True       # toggle with Z
    blur_in_cycle   = True       # BLUR appears in mode cycle

    focus_mode      = False
    focus_box       = None
    selecting_focus = False

    ar_mode             = False
    show_tracking_frame = False

    fps_counter = 0
    fps_display = 0
    fps_timer   = time.time()

    mp_selfie = mp.solutions.selfie_segmentation

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            raw_h, raw_w = raw_frame.shape[:2]
            frame = cv2.flip(raw_frame, 1)
            h, w  = frame.shape[:2]

            # ── FPS ───────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer   = time.time()

            # ── Hand tracking ─────────────────────────────────────────
            hands = hand_tracker.find_hands(frame)

            triggered_gesture = None

            # ── Pose tracking (every 2nd frame) ───────────────────────
            landmarks = pose_tracker.update(frame, every_n=2)

            # ── Draw mode: handle gestures ────────────────────────────
            if canvas.draw_mode:
                pointing = gesture.pointing_hand(hands)
                if pointing is not None:
                    tip = pointing.landmark[8]
                    canvas.pen_at(int(tip.x * w), int(tip.y * h))
                elif gesture.fist_any(hands):
                    canvas.pen_up()

            else:
                # ── Mode cycle (TWO-hand pinch only) ──────────────────
                if gesture.two_hand_pinch_triggered(hands):
                    mode_idx       = (mode_idx + 1) % len(MODES)
                    invisible_mode = MODES[mode_idx]
                    renderer.push_toast(f"Mode  {invisible_mode}")
                    triggered_gesture = "PINCH"

                # ── Focus selection (peace) ────────────────────────────
                if gesture.peace_triggered(hands):
                    if not selecting_focus:
                        selecting_focus = True
                        focus_points.clear()
                        renderer.push_toast("Draw Focus Region — 4 clicks")
                    triggered_gesture = "PEACE V"

                if triggered_gesture:
                    renderer.flash_gesture()

                # ── Zoom (one-hand pinch, continuous) ─────────────────
                if zoom_enabled:
                    zoom_level = zoom_ctrl.update(hands)
                else:
                    zoom_level = 1.0

            # ── Portal detection ──────────────────────────────────────
            portal = portal_detector.detect(hands, frame.shape)

            # ── Segmentation (only when needed) ──────────────────────
            need_mask = (invisible_mode in ("FULL", "BLUR")
                         or (invisible_mode == "PORTAL" and portal is not None)
                         or custom_bg.enabled)

            if need_mask:
                hard_mask, soft_mask = make_person_mask(segmenter, frame)
            else:
                hard_mask = np.zeros((h, w), dtype=np.uint8)
                soft_mask = np.zeros((h, w), dtype=np.float32)

            # ── Resolve background ────────────────────────────────────
            use_custom = custom_bg.enabled
            active_bg  = background

            if use_custom:
                cb = custom_bg.get_frame(frame.shape)
                if cb is not None:
                    active_bg = cb

            # ── Composite output ──────────────────────────────────────
            active_portal = None

            if use_custom:
                # Custom BG = virtual background (person visible, BG replaced)
                output = apply_virtual_background(frame, active_bg, soft_mask)

            elif invisible_mode == "FULL":
                output = apply_full_invisibility(frame, active_bg, soft_mask)

            elif invisible_mode == "BLUR":
                output = apply_background_blur(frame, soft_mask, blur_strength)

            else:   # PORTAL
                output = apply_portal_invisibility(
                    frame, active_bg, portal, soft_mask
                )
                active_portal = portal

            # ── Focus window ──────────────────────────────────────────
            if selecting_focus and len(focus_points) == 4:
                xs = [p[0] for p in focus_points]
                ys = [p[1] for p in focus_points]
                focus_box       = (min(xs), min(ys), max(xs), max(ys))
                selecting_focus = False
                focus_mode      = True
                renderer.push_toast("Focus region set")

            if focus_mode and focus_box:
                output = apply_focus_window(output, active_bg, focus_box)

            # ── AR overlay ────────────────────────────────────────────
            if ar_mode:
                raw_corners = target_tracker.detect(raw_frame)
                corners     = flip_corners_for_mirror(raw_corners, raw_w)
                output      = ar_renderer_obj.draw_target_overlay(output, corners)
                if show_tracking_frame:
                    output = ar_renderer_obj.draw_tracking_frame(output, corners)

            # ── Hand skeleton ─────────────────────────────────────────
            if not canvas.draw_mode:
                output = renderer.draw_hands(output, hands,
                                             mp_hands_mod, mp_draw)

            # ── Portal visual ─────────────────────────────────────────
            if invisible_mode == "PORTAL" and not use_custom:
                output = renderer.draw_portal(output, active_portal)

            # ── Focus selection dots ───────────────────────────────────
            if selecting_focus and focus_points:
                output = renderer.draw_focus_points(output, focus_points)

            # ── AR Canvas drawing ─────────────────────────────────────
            output = canvas.render(output, pose_tracker)

            # ── Zoom ──────────────────────────────────────────────────
            if zoom_enabled:
                zoom_level = zoom_ctrl.level
                if zoom_level > 1.01:
                    output = ZoomController.apply(output, zoom_level)
            else:
                zoom_level = 1.0

            # ── HUD ───────────────────────────────────────────────────
            renderer.tick()
            output = renderer.draw_hud(
                output, invisible_mode, active_portal,
                fps            = fps_display,
                ar_mode        = ar_mode,
                focus_mode     = focus_mode,
                focus_box      = focus_box,
                selecting      = selecting_focus,
                gesture_name   = triggered_gesture,
                zoom_level     = zoom_level,
                vcam_active    = vcam.enabled,
                custom_bg_name = custom_bg.source_name,
                draw_mode      = canvas.draw_mode,
                canvas         = canvas,
            )

            cv2.imshow("RealityFrame", output)
            vcam.send(output)

            # ── Keyboard controls ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("b"):
                renderer.push_toast("Recapturing background…")
                background_model.capture(cap, frames_count=180)
                background = background_model.get()
                invalidate_bg_cache()
                renderer.push_toast("Background updated")

            elif key == ord("f"):
                if not canvas.draw_mode:
                    if focus_box is not None:
                        focus_mode = not focus_mode
                        renderer.push_toast(
                            "Focus ON" if focus_mode else "Focus OFF")
                    else:
                        renderer.push_toast("No focus region — press R first")

            elif key == ord("r"):
                if not canvas.draw_mode:
                    focus_points.clear()
                    focus_box       = None
                    focus_mode      = False
                    selecting_focus = True
                    renderer.push_toast("Click 4 corners to set focus region")

            elif key == ord("a"):
                if not canvas.draw_mode:
                    ar_mode = not ar_mode
                    renderer.push_toast("AR ON" if ar_mode else "AR OFF")

            elif key == ord("v"):
                show_tracking_frame = not show_tracking_frame

            # ── Virtual Camera ────────────────────────────────────────
            elif key == ord("c"):
                if canvas.draw_mode:
                    canvas.cycle_color()
                    renderer.push_toast(f"Colour  {canvas.color_name}")
                else:
                    enabled = vcam.toggle()
                    renderer.push_toast(
                        "Virtual Cam ON" if enabled else "Virtual Cam OFF")

            # ── Custom Background ─────────────────────────────────────
            elif key == ord("i"):
                if not _bg_prompt_active and not canvas.draw_mode:
                    _bg_prompt_active = True
                    def _run():
                        global _bg_prompt_active
                        _open_bg_picker(custom_bg, renderer)
                        _bg_prompt_active = False
                    threading.Thread(target=_run, daemon=True).start()

            # ── Draw Mode ────────────────────────────────────────────
            elif key == ord("d"):
                canvas.toggle_draw_mode()
                renderer.push_toast(
                    "Draw Mode ON — point finger to draw"
                    if canvas.draw_mode
                    else "Draw Mode OFF"
                )

            # ── Draw mode: brush size (W) ────────────────────────────
            elif key == ord("w"):
                if canvas.draw_mode:
                    canvas.cycle_brush()
                    renderer.push_toast(f"Brush  {canvas.brush_size}px")

            # ── Draw mode: anchor to body (N) ────────────────────────
            elif key == ord("n"):
                if canvas.draw_mode:
                    canvas.pen_up()
                    ok = canvas.anchor_to_body(pose_tracker)
                    renderer.push_toast(
                        "Sticker anchored to body!" if ok
                        else "No body detected — stand in frame"
                    )

            # ── Draw mode: undo (U) ───────────────────────────────────
            elif key == ord("u"):
                if canvas.draw_mode:
                    canvas.undo()

            # ── Draw mode: clear (X) ──────────────────────────────────
            elif key == ord("x"):
                if canvas.draw_mode:
                    canvas.clear_all()
                    renderer.push_toast("Canvas cleared")

            # ── Blur strength ─────────────────────────────────────────
            elif key == ord("["):
                blur_strength = max(5, blur_strength - 10)
                renderer.push_toast(f"Blur  {blur_strength}")

            elif key == ord("]"):
                blur_strength = min(101, blur_strength + 10)
                renderer.push_toast(f"Blur  {blur_strength}")

            # ── Zoom toggle ───────────────────────────────────────────
            elif key == ord("z"):
                if zoom_ctrl.level > 1.05:
                    zoom_ctrl.reset()
                    renderer.push_toast("Zoom reset  1.0×")
                else:
                    zoom_enabled = not zoom_enabled
                    renderer.push_toast(
                        "Zoom enabled" if zoom_enabled else "Zoom disabled")

    cap.release()
    pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()