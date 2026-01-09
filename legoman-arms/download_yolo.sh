#!/bin/bash
# Download YOLO-tiny files for object detection

echo "Downloading YOLOv3-tiny weights (8.9 MB)..."
wget -q --show-progress https://pjreddie.com/media/files/yolov3-tiny.weights

echo "Downloading YOLOv3-tiny config..."
wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg

echo "Downloading COCO class names..."
wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

echo "Done! YOLO files downloaded."
ls -lh yolov3-tiny.* coco.names
