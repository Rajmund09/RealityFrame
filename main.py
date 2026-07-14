import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog
import argparse

from core.background import BackgroundModel
from core.custom_background import CustomBackground
from core.virtual_cam import VirtualCamOutput
from core.video_capture import ThreadedVideoCapture
from core.zoom import ZoomController
from core.ar_canvas import ARCanvas
from core.compositor import (
    make_person_mask, invalidate_bg_cache,
    apply_virtual_background, apply_full_invisibility,
    apply_portal_invisibility, apply_background_blur,
    apply_focus_window
)
from vision.hand_tracker import HandTracker
from vision.portal_detector import PortalDetector
from vision.gesture import GestureController
from vision.pose_tracker import PoseTracker
from graphics.renderer import Renderer

from ar.target_tracker import TargetTracker
from ar.overlay_factory import OverlayFactory
from ar.ar_renderer import ARRenderer


def flip_corners_for_mirror(corners, frame_width):
    if corners is None:
        return None
    flipped = corners.copy()
    flipped[:, 0] = frame_width - flipped[:, 0]
    flipped = np.array([flipped[1], flipped[0], flipped[3], flipped[2]],
                       dtype=np.float32)
    return flipped

class RealityFrameApp:
    def __init__(self, camera_index=0):
        self.focus_points = []
        self.MODES = ["PORTAL", "FULL", "BLUR"]
        self.mode_idx = 0
        self.invisible_mode = self.MODES[self.mode_idx]
        
        self.blur_strength = 31
        self.zoom_enabled = True
        
        self.focus_mode = False
        self.focus_box = None
        self.selecting_focus = False
        
        self.ar_mode = False
        self.show_tracking_frame = False
        
        self.fps_counter = 0
        self.fps_display = 0
        self.fps_timer = time.time()
        
        self.bg_prompt_active = False

        self.cap = ThreadedVideoCapture(camera_index)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup window
        cv2.namedWindow("RealityFrame")
        cv2.setMouseCallback("RealityFrame", self.mouse_callback)

        # Components
        self.background_model = BackgroundModel()
        self.background_model.capture(self.cap, frames_count=180)
        self.background = self.background_model.get()

        self.hand_tracker = HandTracker()
        self.portal_detector = PortalDetector()
        self.gesture = GestureController()
        self.renderer = Renderer()
        self.zoom_ctrl = ZoomController()
        self.custom_bg = CustomBackground()
        self.vcam = VirtualCamOutput(self.actual_w, self.actual_h, fps=30)
        self.canvas = ARCanvas()
        self.pose_tracker = PoseTracker()

        self.target_tracker = TargetTracker(marker_id=0)
        self.ar_overlay = OverlayFactory.creeper_face(size=700)
        self.ar_renderer_obj = ARRenderer(self.ar_overlay)

        self.mp_hands_mod = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.focus_points) < 4:
                self.focus_points.append((x, y))

    def _open_bg_picker(self):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filetypes = [
            ("All supported", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.mp4 *.avi *.mov *.mkv *.webm"),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select Background Image or Video", filetypes=filetypes)
        root.destroy()

        if path:
            ok = self.custom_bg.set_source(path)
            invalidate_bg_cache()
            self.renderer.push_toast(f"BG: {self.custom_bg.source_name}" if ok else "Could not load file")
        else:
            self.renderer.push_toast("No file selected")
            
        self.bg_prompt_active = False

    def run(self):
        mp_selfie = mp.solutions.selfie_segmentation
        with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
            while True:
                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    time.sleep(0.01)
                    continue
                
                raw_h, raw_w = raw_frame.shape[:2]
                frame = cv2.flip(raw_frame, 1)
                h, w = frame.shape[:2]

                self.fps_counter += 1
                if time.time() - self.fps_timer >= 1.0:
                    self.fps_display = self.fps_counter
                    self.fps_counter = 0
                    self.fps_timer = time.time()

                hands = self.hand_tracker.find_hands(frame)
                triggered_gesture = None
                landmarks = self.pose_tracker.update(frame, every_n=2)

                if self.canvas.draw_mode:
                    pointing = self.gesture.pointing_hand(hands)
                    if pointing is not None:
                        tip = pointing.landmark[8]
                        self.canvas.pen_at(int(tip.x * w), int(tip.y * h))
                    elif self.gesture.fist_any(hands):
                        self.canvas.pen_up()
                else:
                    if self.gesture.two_hand_pinch_triggered(hands):
                        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
                        self.invisible_mode = self.MODES[self.mode_idx]
                        self.renderer.push_toast(f"Mode  {self.invisible_mode}")
                        triggered_gesture = "PINCH"

                    if self.gesture.peace_triggered(hands):
                        if not self.selecting_focus:
                            self.selecting_focus = True
                            self.focus_points.clear()
                            self.renderer.push_toast("Draw Focus Region - 4 clicks")
                        triggered_gesture = "PEACE V"

                    if triggered_gesture:
                        self.renderer.flash_gesture()

                    if self.zoom_enabled:
                        zoom_level = self.zoom_ctrl.update(hands)
                    else:
                        zoom_level = 1.0

                portal = self.portal_detector.detect(hands, frame.shape)

                need_mask = (self.invisible_mode in ("FULL", "BLUR")
                             or (self.invisible_mode == "PORTAL" and portal is not None)
                             or self.custom_bg.enabled)

                if need_mask:
                    hard_mask, soft_mask = make_person_mask(segmenter, frame)
                else:
                    hard_mask = np.zeros((h, w), dtype=np.uint8)
                    soft_mask = np.zeros((h, w), dtype=np.float32)

                use_custom = self.custom_bg.enabled
                active_bg = self.background

                if use_custom:
                    cb = self.custom_bg.get_frame(frame.shape)
                    if cb is not None:
                        active_bg = cb

                active_portal = None
                if use_custom:
                    output = apply_virtual_background(frame, active_bg, soft_mask)
                elif self.invisible_mode == "FULL":
                    output = apply_full_invisibility(frame, active_bg, soft_mask)
                elif self.invisible_mode == "BLUR":
                    output = apply_background_blur(frame, soft_mask, self.blur_strength)
                else:
                    output = apply_portal_invisibility(frame, active_bg, portal, soft_mask)
                    active_portal = portal

                if self.selecting_focus and len(self.focus_points) == 4:
                    xs = [p[0] for p in self.focus_points]
                    ys = [p[1] for p in self.focus_points]
                    self.focus_box = (min(xs), min(ys), max(xs), max(ys))
                    self.selecting_focus = False
                    self.focus_mode = True
                    self.renderer.push_toast("Focus region set")

                if self.focus_mode and self.focus_box:
                    output = apply_focus_window(output, active_bg, self.focus_box)

                if self.ar_mode:
                    raw_corners = self.target_tracker.detect(raw_frame)
                    corners = flip_corners_for_mirror(raw_corners, raw_w)
                    if corners is not None:
                        output = self.ar_renderer_obj.draw_target_overlay(output, corners)
                        if self.show_tracking_frame:
                            output = self.ar_renderer_obj.draw_tracking_frame(output, corners)

                if not self.canvas.draw_mode:
                    output = self.renderer.draw_hands(output, hands, self.mp_hands_mod, self.mp_draw)

                if self.invisible_mode == "PORTAL" and not use_custom:
                    output = self.renderer.draw_portal(output, active_portal)

                if self.selecting_focus and self.focus_points:
                    output = self.renderer.draw_focus_points(output, self.focus_points)

                output = self.canvas.render(output, self.pose_tracker)

                if self.zoom_enabled:
                    zoom_level = self.zoom_ctrl.level
                    if zoom_level > 1.01:
                        output = ZoomController.apply(output, zoom_level)
                else:
                    zoom_level = 1.0

                self.renderer.tick()
                output = self.renderer.draw_hud(
                    output, self.invisible_mode, active_portal,
                    fps=self.fps_display, ar_mode=self.ar_mode, focus_mode=self.focus_mode,
                    focus_box=self.focus_box, selecting=self.selecting_focus,
                    gesture_name=triggered_gesture, zoom_level=zoom_level,
                    vcam_active=self.vcam.enabled, custom_bg_name=self.custom_bg.source_name,
                    draw_mode=self.canvas.draw_mode, canvas=self.canvas,
                )

                cv2.imshow("RealityFrame", output)
                self.vcam.send(output)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("b"):
                    self.renderer.push_toast("Recapturing background...")
                    self.background_model.capture(self.cap, frames_count=180)
                    self.background = self.background_model.get()
                    invalidate_bg_cache()
                    self.renderer.push_toast("Background updated")
                elif key == ord("f"):
                    if not self.canvas.draw_mode:
                        if self.focus_box is not None:
                            self.focus_mode = not self.focus_mode
                            self.renderer.push_toast("Focus ON" if self.focus_mode else "Focus OFF")
                        else:
                            self.renderer.push_toast("No focus region - press R first")
                elif key == ord("r"):
                    if not self.canvas.draw_mode:
                        self.focus_points.clear()
                        self.focus_box = None
                        self.focus_mode = False
                        self.selecting_focus = True
                        self.renderer.push_toast("Click 4 corners to set focus region")
                elif key == ord("a"):
                    if not self.canvas.draw_mode:
                        self.ar_mode = not self.ar_mode
                        self.renderer.push_toast("AR ON" if self.ar_mode else "AR OFF")
                elif key == ord("v"):
                    self.show_tracking_frame = not self.show_tracking_frame
                elif key == ord("c"):
                    if self.canvas.draw_mode:
                        self.canvas.cycle_color()
                        self.renderer.push_toast(f"Colour  {self.canvas.color_name}")
                    else:
                        enabled = self.vcam.toggle()
                        self.renderer.push_toast("Virtual Cam ON" if enabled else "Virtual Cam OFF")
                elif key == ord("i"):
                    if not self.bg_prompt_active and not self.canvas.draw_mode:
                        self.bg_prompt_active = True
                        threading.Thread(target=self._open_bg_picker, daemon=True).start()
                elif key == ord("d"):
                    self.canvas.toggle_draw_mode()
                    self.renderer.push_toast("Draw Mode ON - point finger to draw" if self.canvas.draw_mode else "Draw Mode OFF")
                elif key == ord("w"):
                    if self.canvas.draw_mode:
                        self.canvas.cycle_brush()
                        self.renderer.push_toast(f"Brush  {self.canvas.brush_size}px")
                elif key == ord("n"):
                    if self.canvas.draw_mode:
                        self.canvas.pen_up()
                        ok = self.canvas.anchor_to_body(self.pose_tracker)
                        self.renderer.push_toast("Sticker anchored to body!" if ok else "No body detected - stand in frame")
                elif key == ord("u"):
                    if self.canvas.draw_mode:
                        self.canvas.undo()
                elif key == ord("x"):
                    if self.canvas.draw_mode:
                        self.canvas.clear_all()
                        self.renderer.push_toast("Canvas cleared")
                elif key == ord("["):
                    self.blur_strength = max(5, self.blur_strength - 10)
                    self.renderer.push_toast(f"Blur  {self.blur_strength}")
                elif key == ord("]"):
                    self.blur_strength = min(101, self.blur_strength + 10)
                    self.renderer.push_toast(f"Blur  {self.blur_strength}")
                elif key == ord("z"):
                    if self.zoom_ctrl.level > 1.05:
                        self.zoom_ctrl.reset()
                        self.renderer.push_toast("Zoom reset  1.0x")
                    else:
                        self.zoom_enabled = not self.zoom_enabled
                        self.renderer.push_toast("Zoom enabled" if self.zoom_enabled else "Zoom disabled")

        self.vcam.close()
        self.cap.release()
        self.pose_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealityFrame AR Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    args = parser.parse_args()

    try:
        app = RealityFrameApp(camera_index=args.camera)
        app.run()
    except RuntimeError as e:
        print(f"Error: {e}")