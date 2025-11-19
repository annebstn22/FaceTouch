# FaceTouch

FaceTouch is an AI-powered project designed to help users become more aware of their face-touching habits and reduce the spread of germs. Using computer vision, it detects when a user is about to touch their face and provides real-time feedback.

## Features

- Detects face-touching gestures using AI and computer vision.
- Provides real-time feedback to discourage face-touching.
- Lightweight and easy to set up using Python and common AI libraries.

## Installation

> **Note:** Do **not** commit your virtual environment to the repository. Use a virtual environment locally.

1. Clone the repository:
```bash
git clone https://github.com/annebstn22/FaceTouch.git
cd FaceTouch

# macOS/Linux
python3 -m venv face_touch_env
source face_touch_env/bin/activate

# Windows
python -m venv face_touch_env
face_touch_env\Scripts\activate

pip install -r requirements.txt

python3 overlay.py
python3 face_touch_kusama.py
