from types import SimpleNamespace as Fake

import numpy as np
import pytest

from common import FrameClock
from posture_monitor import SlouchTimer, posture_height


def test_notification_requires_more_than_five_seconds_and_rearms():
    timer = SlouchTimer()
    assert not timer.update(True, 10)
    assert not timer.update(True, 15)
    assert timer.update(True, 15.01)
    assert not timer.update(True, 30)
    assert timer.alert
    timer.update(False, 31)  # recovery or missing/low-confidence pose
    assert not timer.alert
    assert not timer.update(True, 32)
    assert not timer.update(True, 37)
    assert timer.update(True, 37.01)


def test_interrupted_slouch_does_not_accumulate():
    timer = SlouchTimer()
    timer.update(True, 0)
    timer.update(False, 4)
    assert not timer.update(True, 10)
    assert not timer.update(True, 14)


def test_posture_ratio_uses_pixels_and_rejects_hidden_landmarks():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pose = [Fake(x=0.5, y=0.25, visibility=0.99) for _ in range(33)]
    pose[11] = Fake(x=0.25, y=0.5, visibility=0.99)
    pose[12] = Fake(x=0.75, y=0.5, visibility=0.99)
    assert posture_height(pose, frame) == pytest.approx(120 / 320)
    pose[7].visibility = 0.1
    assert posture_height(pose, frame) is None
    pose[7].visibility = 0.99
    pose[12].x = 0.25
    assert posture_height(pose, frame) is None


def test_frame_clock_increases_when_clock_does_not(monkeypatch):
    monkeypatch.setattr("common.time.monotonic", lambda: 10.0)
    clock = FrameClock()
    assert [clock.now_ms() for _ in range(3)] == [10000, 10001, 10002]


def test_menu_controls_start_calibrate_stop_and_quit(monkeypatch):
    import posture_monitor as module

    class MenuItem:
        def __init__(self, title, callback=None):
            self.title, self.callback = title, callback

        def set_callback(self, callback):
            self.callback = callback

    class App:
        def __init__(self, *args, **kwargs):
            pass

    class Timer:
        def __init__(self, callback, interval):
            self.active = False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

    class Worker(module.MonitorWorker):
        active = False

        @property
        def running(self):
            return self.active

        def start(self):
            self.active = True
            self.stop_requested.clear()

    quit_calls = []
    monkeypatch.setattr(module, "MonitorWorker", Worker)
    ui = Fake(App=App, MenuItem=MenuItem, Timer=Timer,
              quit_application=lambda: quit_calls.append(True),
              alert=lambda *args: None)
    app = module.create_menu_app(ui)
    assert not app.worker.running
    assert app.calibrate_item.callback is None
    app.toggle(None)
    assert app.worker.running
    assert app.start_item.title == "Stop recording"
    app.calibrate(None)
    assert app.worker.calibration_requested.is_set()
    app.toggle(None)
    assert app.worker.stop_requested.is_set()
    assert app.start_item.callback is None
    app.worker.active = False
    app.worker.messages.put(("stopped", "Camera off"))
    app.poll(None)
    assert app.start_item.title == "Start recording"
    assert app.calibrate_item.callback is None
    app.quit(None)
    app.poll(None)
    assert quit_calls == [True]
    assert not app.poller.active


def test_notification_uses_native_macos_delivery(monkeypatch):
    import posture_monitor as module
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    module.notify_posture()
    args, kwargs = calls[0]
    assert args[0][0] == "/usr/bin/osascript"
    assert "display notification" in args[0][2]
    assert "Fix your posture" in args[0][2]
    assert kwargs["check"] and kwargs["timeout"] == 5
