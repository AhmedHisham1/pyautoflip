# PyAutoFlip

A Python library for saliency-aware video cropping that automatically reframes videos to different aspect ratios while preserving important content.

**Note**: This is a Python implementation inspired by [MediaPipe's AutoFlip](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md). The original MediaPipe AutoFlip solution is no longer actively supported, so this project provides a maintained alternative using similar techniques.

## What it does

PyAutoFlip analyzes videos to identify salient content (faces, objects, motion) and intelligently crops frames to fit target aspect ratios. This is useful for adapting content between different platforms (e.g., landscape videos for portrait social media formats).

## Installation

```bash
# From PyPI
pip install pyautoflip
```

## Quick Start

### Command Line

```bash
# Convert a landscape video to portrait (9:16) 
pyautoflip reframe -i input.mp4 -o output.mp4

# Convert to square format
pyautoflip reframe -i input.mp4 -o output.mp4 --aspect-ratio 1:1

# Enable debug visualizations
pyautoflip reframe -i input.mp4 -o output.mp4 --debug
```

### Python API

```python
from pyautoflip import reframe_video

# Basic usage
reframe_video(
    input_path="input.mp4",
    output_path="output.mp4", 
    target_aspect_ratio="9:16"
)

# With options
reframe_video(
    input_path="input.mp4",
    output_path="output.mp4",
    target_aspect_ratio="1:1",
    motion_threshold=0.3,        # Lower = more stable crops
    padding_method="blur",       # or "solid_color"
    debug_mode=True
)
```

## How it works

1. **Scene Detection**: Identifies scene boundaries in the video
2. **Content Analysis**: Detects faces (InsightFace) and objects (MediaPipe) 
3. **Saliency-aware Cropping**: Determines optimal crop regions based on detected content
4. **Temporal Smoothing**: Applies smooth camera motion or stable crops as appropriate

## Options

- `--aspect-ratio`: Target aspect ratio (e.g., "9:16", "1:1", "4:3")
- `--motion-threshold`: Camera motion sensitivity (0.0 = very stable, 1.0 = allow more motion)
- `--padding-method`: How to handle content that doesn't fit ("blur" or "solid_color")
- `--debug`: Enable debug mode with visualizations and detailed logging

## Requirements

- Python 3.10+
- FFmpeg (for video processing)

### System dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install ffmpeg
```

## Development

```bash
git clone https://github.com/AhmedHisham1/pyautoflip.git
cd pyautoflip
poetry install
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [MediaPipe AutoFlip](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md) for the original concept and methodology
- [InsightFace](https://github.com/deepinsight/insightface) for face detection
- [MediaPipe](https://github.com/google/mediapipe) for object detection
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene analysis
