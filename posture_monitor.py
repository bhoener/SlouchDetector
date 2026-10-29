"""macOS menu bar posture monitor. Run: python posture_monitor.py [--camera 1].

Start recording enables live analysis only; no video is saved. Calibrate while
sitting upright, facing a fixed eye-level camera with ears/shoulders visible.
A 25% reduction in relative ear height is a slouch heuristic, not a reliable
measurement of forward depth. Looking down can also trigger it.
"""

from __future__ import annotations

import math
import queue
import subprocess
import sys
import threading
import time

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import FrameClock, model_path, open_camera, to_mp_image, to_px

SLOUCH_SECONDS = 5.0
HEIGHT_FRACTION = 0.75


def posture_height(pose, frame) -> float | None:
    """Pixel ear/shoulder height divided by pixel shoulder width, or unknown."""
    ids = (7, 8, 11, 12)  # left/right ears, left/right shoulders
    for i in ids:
        lm = pose[i]
        if (lm.visibility is None or lm.visibility < 0.6
                or not (0 <= lm.x <= 1 and 0 <= lm.y <= 1)):
            return None
    left_ear, right_ear, left_shoulder, right_shoulder = [to_px(pose[i], frame) for i in ids]
    width = math.dist(left_shoulder, right_shoulder)
    if width < 30:
        return None
    ear_y = (left_ear[1] + right_ear[1]) / 2
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    return (shoulder_y - ear_y) / width


class SlouchTimer:
    def __init__(self):
        self.started = None
        self.alert = False

    def update(self, slouched: bool, now: float) -> bool:
        """Return True once per continuous episode lasting more than 5 seconds."""
        if not slouched:
            self.started = None
            self.alert = False
            return False
        if self.started is None:
            self.started = now
        if not self.alert and now - self.started > SLOUCH_SECONDS:
            self.alert = True
            return True
        return False


def notify_posture() -> None:
    """Deliver a native macOS notification from the background worker."""
    subprocess.run(
        ["/usr/bin/osascript", "-e",
         'display notification "Sit upright and relax your shoulders." '
         'with title "Fix your posture" sound name "Glass"'],
        check=True, capture_output=True, text=True, timeout=5,
    )


class MonitorWorker:
    """Own the camera/model on one thread; communicate with UI using a queue."""

    def __init__(self):
        self.stop_requested = threading.Event()
        self.calibration_requested = threading.Event()
        self.messages = queue.Queue()
        self.thread = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running:
            return
        self.stop_requested.clear()
        self.calibration_requested.clear()
        self.thread = threading.Thread(target=self.run, name="posture-camera", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_requested.set()

    def run(self):
        cap = None
        last_status = None

        def status(message):
            nonlocal last_status
            if message != last_status:
                self.messages.put(("status", message))
                last_status = message

        try:
            status("Starting camera...")
            base = mp_python.BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task"))
            options = vision.PoseLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.VIDEO, num_poses=1,
            )
            clock, timer = FrameClock(), SlouchTimer()
            baseline = None
            cap = open_camera()
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                while not self.stop_requested.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera stopped delivering frames. Try Start recording again.")
                    frame = cv2.flip(frame, 1)
                    result = landmarker.detect_for_video(to_mp_image(frame), clock.now_ms())
                    height = (posture_height(result.pose_landmarks[0], frame)
                              if result.pose_landmarks else None)
                    now = time.monotonic()
                    if self.calibration_requested.is_set():
                        self.calibration_requested.clear()
                        if height is not None and height > 0:
                            baseline = height
                            timer.update(False, now)
                        else:
                            self.messages.put(("error", "Calibration failed: face the camera with both ears "
                                                  "and shoulders visible, then try Calibrate again."))
                    slouched = (baseline is not None and height is not None
                                and height < baseline * HEIGHT_FRACTION)
                    if timer.update(slouched, now) and not self.stop_requested.is_set():
                        try:
                            notify_posture()
                        except (OSError, subprocess.SubprocessError) as exc:
                            self.messages.put(("error", f"Could not send notification: {exc}"))
                    if height is None:
                        status("Show both ears and shoulders")
                    elif baseline is None:
                        status("Sit upright, then choose Calibrate")
                    elif timer.alert:
                        status("Fix your posture")
                    elif slouched:
                        status("Slouch detected (waiting 5 seconds)")
                    else:
                        status("Posture OK")
        except (Exception, SystemExit) as exc:
            self.messages.put(("error", str(exc)))
        finally:
            if cap is not None:
                cap.release()
            self.messages.put(("stopped", "Camera off"))


def create_menu_app(rumps):
    """Keep Cocoa imports out of the platform-independent detection/tests."""
    class PostureApp(rumps.App):
        def __init__(self):
            super().__init__("Posture", title="Posture", quit_button=None)
            self.worker = MonitorWorker()
            self.quitting = False
            self.status_item = rumps.MenuItem("Camera off")
            self.start_item = rumps.MenuItem("Start recording", callback=self.toggle)
            self.calibrate_item = rumps.MenuItem("Calibrate upright posture")
            self.menu = [self.status_item, None, self.start_item,
                         self.calibrate_item, None,
                         rumps.MenuItem("Quit", callback=self.quit)]
            self.poller = rumps.Timer(self.poll, 0.2)
            self.poller.start()

        def toggle(self, _):
            if self.worker.running:
                self.worker.stop()
                self.start_item.set_callback(None)
                self.calibrate_item.set_callback(None)
                self.status_item.title = "Stopping camera..."
            else:
                self.worker.start()
                self.start_item.title = "Stop recording"
                self.calibrate_item.set_callback(self.calibrate)

        def calibrate(self, _):
            if self.worker.running:
                self.worker.calibration_requested.set()

        def quit(self, _):
            self.quitting = True
            self.worker.stop()
            self.start_item.set_callback(None)
            self.calibrate_item.set_callback(None)
            self.status_item.title = "Closing camera..."

        def poll(self, _):
            while True:
                try:
                    kind, message = self.worker.messages.get_nowait()
                except queue.Empty:
                    break
                if kind == "error" and not self.quitting:
                    rumps.alert("Posture monitor", message)
                elif kind == "status" and not self.worker.stop_requested.is_set():
                    self.status_item.title = message
                elif kind == "stopped":
                    self.status_item.title = message
            if not self.worker.running:
                if self.quitting:
                    self.poller.stop()
                    rumps.quit_application()
                else:
                    self.start_item.title = "Start recording"
                    self.start_item.set_callback(self.toggle)
                    self.calibrate_item.set_callback(None)

    return PostureApp()


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("The menu bar posture monitor requires macOS.")
    import rumps
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app = create_menu_app(rumps)
    try:
        app.run()
    finally:
        app.worker.stop()
        if app.worker.thread is not None:
            app.worker.thread.join(timeout=5)


if __name__ == "__main__":
    main()
