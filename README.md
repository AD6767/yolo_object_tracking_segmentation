# YOLO Object Detection, Tracking, Segmentation

Small collection of demos using **Ultralytics YOLOv8** for:
- object detection on images/videos
- multi-object tracking + trails
- basic object counting
- instance segmentation

## Project layout
- `data/` - sample inputs (images/videos)
- `simple_object_detection.py` - image detection
- `multi_object_from_video.py` - video detection/tracking
- `people_with_trail.py` - person tracking with trail
- `object_counting.py` - counting
- `segmentation.py` - instance segmentation
- `live_camera_feed.py` - live feed tracking
- `requirements.txt`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Notes: weights are not committed, Ultralytics can usually auto-download them when you run the scripts.

## Run demos
* Image detection from single frame: `python simple_object_detection.py`
* multi-object detection from video: `python multi_object_from_video.py`
* Live camera feed object detection: `python live_camera_feed.py`
* People tracking with trail: `python people_with_trail.py`
* Object counting: `python object_counting.py`
* Segmentation: `python segmentation.py`

## Inputs
Please add sample inputs into `data/`:
* data/sample1.jpeg
* data/sample_video.mp4
* data/sample_video_2.mp4

## Credits
Built with the Ultralytics YOLOv8 ecosystem
