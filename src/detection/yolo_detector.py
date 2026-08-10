import os
import numpy as np
import torch
from PIL import Image
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class YOLOFaceDetector:

    def __init__(self, config_path='configs/config.yaml'):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError('ultralytics not installed. Run:\n  pip install ultralytics')
        config = load_config(config_path)
        yolo_cfg = config.get('yolo_detector', {})
        self.conf_threshold = yolo_cfg.get('conf_threshold', 0.5)
        self.iou_threshold = yolo_cfg.get('iou_threshold', 0.45)
        self.img_size = yolo_cfg.get('img_size', 640)
        weights_path = yolo_cfg.get('weights', 'models/yolo/best.pt')
        pretrained = yolo_cfg.get('pretrained', 'yolov8n.pt')
        if os.path.exists(weights_path):
            print(f'Loading weights from {weights_path}')
            self.model = YOLO(weights_path)
        else:
            print(f'Weights not found. Using pretrained {pretrained}.')
            self.model = YOLO(pretrained)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device: {self.device}')

    def detect(self, image):
        results = self.model.predict(source=image, conf=self.conf_threshold, iou=self.iou_threshold, imgsz=self.img_size, device=self.device, verbose=False)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            print('No faces detected.')
            return (None, None)
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        print(f'Faces detected: {len(boxes)}')
        return (boxes, confs)

    def crop_faces(self, pil_image, boxes, pad=10):
        if boxes is None:
            return []
        crops = []
        for box in boxes:
            (x1, y1, x2, y2) = [int(c) for c in box]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(pil_image.width, x2 + pad)
            y2 = min(pil_image.height, y2 + pad)
            crop = pil_image.crop((x1, y1, x2, y2))
            crops.append(crop)
        print(f'Cropped faces: {len(crops)}')
        return crops

    def detect_and_crop(self, pil_image, pad=10):
        (boxes, confs) = self.detect(pil_image)
        crops = self.crop_faces(pil_image, boxes, pad=pad)
        return (boxes, confs, crops)
