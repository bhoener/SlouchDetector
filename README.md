# Vibe-Coding Computer Vision: three mini demos

Starter code, prompts and an AI skill for the OSU AI Club's computer-vision
workshop. Each demo is a small webcam app built on Google's
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) and
vibe-coded with an AI assistant.

| Demo | MediaPipe task | Starter | Finished version | Prompts |
|---|---|---|---|---|
| 1. Air drawing | Hand Landmarker | [starter/air_drawing.py](starter/air_drawing.py) | [solutions/air_drawing.py](solutions/air_drawing.py) | [01-air-drawing.md](prompts/01-air-drawing.md) |
| 2. Rock-paper-scissors vs. the computer | Gesture Recognizer | [starter/rock_paper_scissors.py](starter/rock_paper_scissors.py) | [solutions/rock_paper_scissors.py](solutions/rock_paper_scissors.py) | [02-rock-paper-scissors.md](prompts/02-rock-paper-scissors.md) |
| 3. Squat or push-up counter | Pose Landmarker | [starter/rep_counter.py](starter/rep_counter.py) | [solutions/rep_counter.py](solutions/rep_counter.py) | [03-rep-counter.md](prompts/03-rep-counter.md) |

## Posture monitor

After setup, run `python posture_monitor.py` (optional: `--camera 1`) on macOS.
A **Posture** item appears in the upper menu bar; there is no video window.

- **Start recording** starts webcam posture analysis in a background thread.
  No video or images are saved. The camera stays off until you start it.
- **Calibrate upright posture** saves a fresh measurement. Face a fixed,
  eye-level camera with both ears and shoulders visible and sit upright first.
  Recalibrate after moving the camera. The menu shows tracking/calibration status.
- **Stop recording** releases the camera. Starting again requires calibration.
- **Quit** stops monitoring, releases the camera, and removes the menu bar item.

A 25% drop in ear-to-shoulder height relative to shoulder width starts a timer.
After more than five continuous seconds, macOS displays **Fix your posture**
with a sound. One notification is sent per continuous episode; recovering or
losing reliable tracking resets the timer. Allow Camera access for your Python
launcher/terminal when macOS asks. Notifications are delivered through Apple's
`osascript`; allow its notification sender in System Settings > Notifications
if prompted. Focus mode and notification settings can suppress banners/sounds.

This uses PoseLandmarker in VIDEO mode and mirrored frames. The frontal image
heuristic uses ears (7/8) and shoulders (11/12); looking down can trigger it,
and forward lean without a visible change in head height may be missed.
The process must remain running; it does not install a login item or service.

## Setup (about 5 minutes)

You need a webcam and Python 3.9–3.13. Python 3.12 is the safe pick on any laptop.

```bash
git clone <this repo's URL>
cd vibe-coding-cv-demos
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: py -3.12 -m venv .venv, then .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py        # the three model files, once
python see_yourself.py           # dots on your hand = you're ready
```

**Intel Mac?** MediaPipe stopped publishing Intel Mac builds after version 0.10.21
(February 2025). With Python 3.9–3.12, pip installs 0.10.21 automatically and
everything here works with it.

**Mac camera permission:** the first run asks to let your terminal app (Terminal,
iTerm or VS Code) use the camera. If the window stays black, turn it on in
System Settings › Privacy & Security › Camera and restart the terminal.

## Vibe-coding a demo

1. Run `python see_yourself.py` to make sure the camera and models work.
2. Open the demo's prompt file. Paste the **from scratch** prompt into any AI chat,
   or open this repo in an AI coding tool and use the **with the starter** prompt.
3. Do the three checks in the prompt file before you run the code.
4. When it breaks, paste the full error back. Each prompt file has follow-ups.
5. Stuck, or short on time? Run the finished version in `solutions/`.

## Give your AI the rules once

- **[AGENTS.md](AGENTS.md):** the rules for this repo: Tasks API only, where the
  models live, the shared helpers, the common bugs. Codex, Cursor and GitHub
  Copilot read it automatically.
- **[CLAUDE.md](CLAUDE.md) and [.claude/skills/mediapipe-tasks](.claude/skills/mediapipe-tasks/SKILL.md):**
  Claude Code loads CLAUDE.md every session and the skill (landmark numbers, demo
  recipes, bug table) when you work on MediaPipe code.
- **Using a chat app?** Paste AGENTS.md into the conversation first.
- **More:** the community [cv-mediapipe skill](https://www.skills.sh/damionrashford/media-os/cv-mediapipe)
  (`npx skills add https://github.com/damionrashford/media-os --skill cv-mediapipe`;
  read any skill before you install it), and Context7, which pulls current library
  docs into a prompt when you add "use context7".

## What's in here

```text
common.py               shared helpers: camera, timestamps, drawing
download_models.py      gets the three model files from Google
see_yourself.py         setup check: your hand as 21 landmarks
models/                 the .task model files go here
starter/                the demos with TODOs for you (or your AI)
solutions/              finished demos, for reference or as a live-demo backup
prompts/                copy-paste prompts, checks and fixes
AGENTS.md, CLAUDE.md    rules for AI coding tools
.claude/skills/         Claude Code skill: mediapipe-tasks
tests/                  python -m pytest tests
```

## Troubleshooting

| You see | Fix |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | That's legacy MediaPipe code, removed in 0.10.31. Ask the AI to use the Tasks API, or give it AGENTS.md. |
| `Missing model file` | `python download_models.py` |
| `Couldn't open the webcam`, or a black window | Close Zoom, Teams or FaceTime. On a Mac, allow camera access for your terminal (see Setup). External webcam: add `--camera 1`. |
| `No matching distribution found for mediapipe` | Unsupported Python version. Use 3.12 (Intel Mac: 3.9–3.12). |
| SSL error in `download_models.py` | On a python.org Mac install, run "Install Certificates.command" in your Python folder, or download the files in a browser (the script prints the links). |
| Odd cv2 errors after installing `opencv-python` | `pip uninstall opencv-python`. MediaPipe already installs `opencv-contrib-python`. |

## Tests

```bash
pip install pytest
python -m pytest tests
```

The tests run every script against a fake webcam and fake MediaPipe results, so
they need no camera or model files. They pass on MediaPipe 1.1.1 (Python 3.9,
3.11, 3.12 and 3.13) and on 0.10.21, the last Intel Mac release (Python 3.12).

## For the presenter

- Download the models before the workshop. To spare members the download on
  venue wifi, you can commit them (`git add models/*.task`).
- Run each finished demo once on the laptop you'll present from.
- Keep the finished versions open as a backup in case a live prompt stalls.

## Sources

- MediaPipe guides: [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python),
  [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer),
  [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).
- [google-ai-edge/mediapipe#6204](https://github.com/google-ai-edge/mediapipe/issues/6204): the Solutions API was removed in 0.10.31.
- Air drawing logic: [Educative, Create a virtual paint app using MediaPipe](https://www.educative.io/answers/create-a-virtual-paint-app-using-mediapipe).
- Rock-paper-scissors lock-in: [andypotato on DEV](https://dev.to/andypotato/rock-paper-scissors-with-hand-gesture-recognition-465g).
- Rep counter thresholds: [Labellerr](https://www.labellerr.com/blog/ai-powered-pushup-counter-form-corrector/),
  [liveTrain on Hugging Face](https://huggingface.co/spaces/pjxcharya/liveTrain/commit/14e399f9849262825b303a03a4152511168afea5),
  [LearnOpenCV AI Fitness Trainer](https://learnopencv.com/ai-fitness-trainer-using-mediapipe).
- AI setup: [Claude Code best practices](https://code.claude.com/docs/en/best-practices.md),
  [GitHub Copilot prompt engineering](https://docs.github.com/en/copilot/concepts/prompting/prompt-engineering).
