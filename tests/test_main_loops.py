"""Run every script's main() against a fake webcam and a fake MediaPipe task.

This exercises the whole loop (detection -> logic -> drawing -> keys)
without a camera or model files.
"""

from __future__ import annotations

import math
import random
import time
from types import SimpleNamespace as Fake

import cv2
import mediapipe as mp
import numpy as np
import pytest
from mediapipe.tasks.python import vision

from conftest import fake_hand, fake_pose, load_module

BACKGROUND = 90
W, H = 1280, 720


class FakeCamera:
    def __init__(self, frames: int):
        self.frames = frames

    def read(self):
        if self.frames == 0:
            return False, None
        self.frames -= 1
        return True, np.full((H, W, 3), BACKGROUND, np.uint8)

    def release(self):
        pass


class FakeTask:
    """Stands in for HandLandmarker, GestureRecognizer and PoseLandmarker."""

    def __init__(self, results):
        self.results = iter(results)
        self.timestamps = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _run(self, image, timestamp_ms):
        assert isinstance(image, mp.Image)
        assert not self.timestamps or timestamp_ms > self.timestamps[-1], "timestamps must increase"
        self.timestamps.append(timestamp_ms)
        return next(self.results)

    detect_for_video = _run
    recognize_for_video = _run


@pytest.fixture
def screen(monkeypatch):
    """Capture shown frames and feed key presses instead of opening windows."""
    state = Fake(shown=[], keys={})
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: state.shown.append(frame.copy()))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay=0: state.keys.get(len(state.shown) - 1, -1))
    return state


def run(module, monkeypatch, task_class, results):
    task = FakeTask(results)
    monkeypatch.setattr(module, "open_camera", lambda: FakeCamera(len(results)))
    monkeypatch.setattr(module, "model_path", lambda name: f"models/{name}")
    monkeypatch.setattr(task_class, "create_from_options", lambda options: task)
    module.main()
    assert len(task.timestamps) == len(results)
    return task


def hand_result(hand):
    return Fake(hand_landmarks=[hand], handedness=[[Fake(category_name="Right", score=0.98)]])


NO_HAND = Fake(hand_landmarks=[], handedness=[])


def test_see_yourself(monkeypatch, screen):
    module = load_module("see_yourself.py")
    results = [hand_result(fake_hand((True, False, False, False)))] * 3 + [NO_HAND] * 2
    run(module, monkeypatch, vision.HandLandmarker, results)
    assert len(screen.shown) == 5


def test_posture_monitor_worker(monkeypatch):
    module = load_module("posture_monitor.py")
    worker = module.MonitorWorker()
    pose = fake_pose(left_visibility=0.99, right_visibility=0.99)
    task = FakeTask([Fake(pose_landmarks=[pose])] * 5)
    camera = FakeCamera(5)
    released, notifications, options_seen = [], [], []
    monkeypatch.setattr(camera, "release", lambda: released.append(True))
    monkeypatch.setattr(module, "open_camera", lambda: camera)
    monkeypatch.setattr(module, "model_path", lambda name: f"models/{name}")

    def create(options):
        options_seen.append(options)
        return task

    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", create)
    heights = iter([0.5, 0.3, 0.3, 0.3, None])
    monkeypatch.setattr(module, "posture_height", lambda pose, frame: next(heights))
    clock = iter(x * 2.0 for x in range(100))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module, "notify_posture", lambda: notifications.append(True))
    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("No video UI allowed"))
    worker.calibration_requested.set()
    worker.run()
    assert notifications == [True]
    assert released == [True]
    assert len(task.timestamps) == 5
    assert options_seen[0].running_mode == vision.RunningMode.VIDEO
    assert options_seen[0].base_options.model_asset_path == "models/pose_landmarker_lite.task"


def test_posture_monitor_releases_camera_on_model_failure(monkeypatch):
    module = load_module("posture_monitor.py")
    camera = FakeCamera(1)
    released = []
    monkeypatch.setattr(camera, "release", lambda: released.append(True))
    monkeypatch.setattr(module, "open_camera", lambda: camera)
    monkeypatch.setattr(module, "model_path", lambda name: f"models/{name}")

    def fail(options):
        raise RuntimeError("Broken model")

    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", fail)
    worker = module.MonitorWorker()
    worker.run()
    assert released == [True]
    messages = list(worker.messages.queue)
    assert ("error", "Broken model") in messages
    assert messages[-1] == ("stopped", "Camera off")


def test_air_drawing_solution(monkeypatch, screen, air):
    painters = []

    class SpyPainter(air.Painter):
        def __init__(self, shape):
            super().__init__(shape)
            painters.append(self)

    monkeypatch.setattr(air, "Painter", SpyPainter)
    draw = [hand_result(fake_hand((True, False, False, False), tip_x=0.2 + 0.02 * i)) for i in range(20)]
    lift = [hand_result(fake_hand((True, True, False, False)))] * 3
    fist = [hand_result(fake_hand((False, False, False, False)))] * (air.CLEAR_HOLD_FRAMES + 2)
    run(air, monkeypatch, vision.HandLandmarker, draw + lift + fist + [NO_HAND] * 3)

    stroke_pixel = (216, 400)  # y = 0.3 * 720, x between the stroke's start and end
    assert screen.shown[19][stroke_pixel].tolist() == list(air.ORANGE)       # stroke is on screen
    assert screen.shown[-1][stroke_pixel].tolist() == [BACKGROUND] * 3       # held fist cleared it
    assert not painters[0].canvas.any()


def test_air_drawing_color_and_clear_keys(monkeypatch, screen, air):
    # Keys are read after a frame is shown: 2 after frame 0, C after frame 5.
    screen.keys = {0: ord("2"), 5: ord("c")}
    draw = [hand_result(fake_hand((True, False, False, False), tip_x=0.2 + 0.03 * i)) for i in range(6)]
    run(air, monkeypatch, vision.HandLandmarker, draw + [NO_HAND])
    on_stroke = (216, int(0.2 * W) + 40)
    assert screen.shown[5][on_stroke].tolist() == list(air.BLUE)    # key 2 = blue pen
    assert screen.shown[6][on_stroke].tolist() == [BACKGROUND] * 3  # C cleared it


def gesture_result(label, score=0.95):
    hand = fake_hand((False, False, False, False))
    return Fake(gestures=[[Fake(category_name=label, score=score)]], hand_landmarks=[hand],
                handedness=[[Fake(category_name="Right", score=0.98)]])


def test_rock_paper_scissors_solution(monkeypatch, screen, rps):
    games = []

    class SpyGame(rps.Game):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            games.append(self)

    clock = iter(x * 0.25 for x in range(100_000))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(random, "choice", lambda options: "scissors")
    monkeypatch.setattr(rps, "Game", SpyGame)
    screen.keys = {0: ord(" ")}  # press SPACE on the first frame

    run(rps, monkeypatch, vision.GestureRecognizer, [gesture_result("Closed_Fist")] * 80)
    game = games[0]
    assert game.state == "over"
    assert game.score == {"player": 3, "computer": 0}  # rock beats scissors every round


def test_rock_paper_scissors_ignores_other_gestures(monkeypatch, screen, rps):
    games = []

    class SpyGame(rps.Game):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            games.append(self)

    clock = iter(x * 0.25 for x in range(100_000))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rps, "Game", SpyGame)
    screen.keys = {0: ord(" ")}
    # About 0.5 fake seconds per frame: 6 frames of countdown, then a 3 second timeout.
    run(rps, monkeypatch, vision.GestureRecognizer, [gesture_result("Thumb_Up")] * 16)
    assert games[0].outcome == "no move"
    assert games[0].score == {"player": 0, "computer": 0}


def pose_with_angle(points, angle_deg):
    """A pose whose angle at points[1] (between points[0] and points[2]) is angle_deg."""
    a_id, b_id, c_id = points
    pose = fake_pose()
    ax, ay, bx, by = 0.40 * W, 0.40 * H, 0.50 * W, 0.50 * H
    start = math.atan2(ay - by, ax - bx)
    end = start + math.radians(angle_deg)
    cx, cy = bx + 150 * math.cos(end), by + 150 * math.sin(end)
    for i, (x, y) in ((a_id, (ax, ay)), (b_id, (bx, by)), (c_id, (cx, cy))):
        pose[i] = Fake(x=x / W, y=y / H, z=0.0, visibility=0.99)
    return Fake(pose_landmarks=[pose], pose_world_landmarks=[], segmentation_masks=None)


def run_reps(rep, monkeypatch, screen, exercise, angles, keys=None):
    counters = []

    class SpyCounter(rep.RepCounter):
        def __init__(self, up, down):
            super().__init__(up, down)
            counters.append(self)

    monkeypatch.setattr(rep, "RepCounter", SpyCounter)
    screen.keys = keys or {}
    points = rep.EXERCISES[exercise]["left"]
    run(rep, monkeypatch, vision.PoseLandmarker, [pose_with_angle(points, a) for a in angles])
    return counters


def test_rep_counter_push_ups(monkeypatch, screen, rep):
    angles = [170, 130, 85, 130, 170, 120, 80, 165, 175, 150]  # two full push-ups
    counters = run_reps(rep, monkeypatch, screen, "push-up", angles)
    assert counters[-1].count == 2


def test_rep_counter_switch_to_squats(monkeypatch, screen, rep):
    angles = [175, 120, 80, 175, 100, 175]  # one full squat (the 100 isn't deep enough)
    counters = run_reps(rep, monkeypatch, screen, "squat", angles, keys={0: ord("s")})
    assert (counters[-1].up, counters[-1].down) == (170, 90)
    assert counters[-1].count == 1


def test_rep_counter_without_a_person(monkeypatch, screen, rep):
    empty = Fake(pose_landmarks=[], pose_world_landmarks=[], segmentation_masks=None)
    run(rep, monkeypatch, vision.PoseLandmarker, [empty] * 3)
    assert len(screen.shown) == 3


@pytest.mark.parametrize("script, task_class, result", [
    ("starter/air_drawing.py", vision.HandLandmarker, hand_result(fake_hand((True, False, False, False)))),
    ("starter/rock_paper_scissors.py", vision.GestureRecognizer, gesture_result("Victory")),
    ("starter/rep_counter.py", vision.PoseLandmarker, pose_with_angle((11, 13, 15), 120)),
])
def test_starters_run_before_the_todos_are_done(monkeypatch, screen, script, task_class, result):
    module = load_module(script)
    run(module, monkeypatch, task_class, [result] * 4)
    assert len(screen.shown) == 4
