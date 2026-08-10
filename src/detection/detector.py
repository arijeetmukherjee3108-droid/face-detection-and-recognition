import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import yaml
import os

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class FaceDetector:

    def __init__(self, config_path='configs/config.yaml'):
        config = load_config(config_path)
        det_cfg = config['detection']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'FaceDetector running on: {self.device}')
        self.model = MTCNN(min_face_size=det_cfg['min_face_size'], thresholds=det_cfg['threshold'], keep_all=True, device=self.device)
        self.image_size = det_cfg['image_size']

    def detect(self, pil_image):
        (boxes, probs, landmarks) = self.model.detect(pil_image, landmarks=True)
        if boxes is None:
            print('No faces detected in this image.')
            return (None, None)
        confidence_threshold = 0.9
        confident_indices = [i for (i, prob) in enumerate(probs) if prob is not None and prob > confidence_threshold]
        if len(confident_indices) == 0:
            print('Faces found but confidence too low.')
            return (None, None)
        boxes = boxes[confident_indices]
        landmarks = landmarks[confident_indices] if landmarks is not None else None
        print(f'Detected {len(boxes)} faces with high confidence.')
        return (boxes, landmarks)

    def crop_faces(self, pil_image, boxes):
        if boxes is None:
            return []
        face_crops = []
        for box in boxes:
            (x1, y1, x2, y2) = [int(coord) for coord in box]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(pil_image.width, x2 + pad)
            y2 = min(pil_image.height, y2 + pad)
            face = pil_image.crop((x1, y1, x2, y2))
            face = face.resize((self.image_size, self.image_size), Image.BILINEAR)
            face_crops.append(face)
        print(f'Cropped {len(face_crops)} faces ready for next stage.')
        return face_crops
