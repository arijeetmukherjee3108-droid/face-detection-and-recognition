import cv2
import numpy as np
from PIL import Image
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')
    pil_image = Image.open(image_path).convert('RGB')
    bgr_image = cv2.imread(image_path)
    return (pil_image, bgr_image)

def draw_boxes(bgr_image, boxes, landmarks=None, color=(0, 255, 0), thickness=2):
    if boxes is None:
        return bgr_image
    image = bgr_image.copy()
    for (i, box) in enumerate(boxes):
        (x1, y1, x2, y2) = [int(c) for c in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if landmarks is not None:
            for (lx, ly) in landmarks[i]:
                cv2.circle(image, (int(lx), int(ly)), radius=3, color=(0, 0, 255), thickness=-1)
    return image

def save_image(bgr_image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr_image)
    print(f'Saved: {save_path}')

def resize_image(pil_image, size):
    return pil_image.resize((size, size), Image.BILINEAR)

def normalize_pixels(pil_image):
    img_array = np.array(pil_image, dtype=np.float32)
    img_array = (img_array - 127.5) / 128.0
    return img_array
