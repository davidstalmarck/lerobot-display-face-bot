# 🤖 Robot Control UI - Quick Start Guide

## Installation

### 1. Install Python Dependencies
```bash
cd /home/david/dev/random/lerobot-display-face-bot/legoman-arms
pip install -r requirements.txt
```

### 2. Install Ollama (Optional - for AI vision)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the vision model (about 4GB)
ollama pull llava
```

## Running the Application

### Start the Server
```bash
cd /home/david/dev/random/lerobot-display-face-bot/legoman-arms
python simple_control_server.py
```

### Open the UI
In another terminal or just open in your browser:
```bash
# Open the UI in browser
xdg-open simple_control_ui.html

# Or navigate to:
# file:///home/david/dev/random/lerobot-display-face-bot/legoman-arms/simple_control_ui.html
```

## Expected Output

When you start the server, you should see:
```
✓ MediaPipe loaded successfully
✓ Ollama is available
✓ LLM analysis thread started
Starting simple control server on port 5001...
Open simple_control_ui.html in your browser
 * Running on http://0.0.0.0:5001
```

## Features

- **Face Detection** - Detects faces with confidence scores
- **Hand Detection** - Detects left/right hands
- **AI Vision** - Natural language scene descriptions via LLaVa
- **Robot Control** - Record, play, stop, and relax buttons
- **Speed Control** - 1x, 1.25x, 1.5x, 2x playback speeds
- **Recording Management** - Name, save, and manage multiple recordings

## Troubleshooting

### Camera not working?
```bash
# Check camera devices
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works!' if cap.isOpened() else 'Camera failed')"
```

### Robot motors not responding?
```bash
# Check USB connection
ls /dev/ttyACM*

# Should show /dev/ttyACM0 (and possibly /dev/ttyACM1)
```

### Ollama not working?
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

The UI will work without Ollama, just without AI descriptions.

## Quick Commands Summary

```bash
# One-time setup
cd /home/david/dev/random/lerobot-display-face-bot/legoman-arms
pip install -r requirements.txt
ollama pull llava  # Optional

# Every time you want to run it
python simple_control_server.py

# Then open simple_control_ui.html in your browser
```

Enjoy! 🎉