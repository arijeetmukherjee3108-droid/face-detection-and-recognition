import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.detection.detector import FaceDetector
from src.classification.classifier import MaskClassifier
from src.recognition.recognizer import FaceRecognizer
from src.utils.image_utils import load_image, save_image
LABEL_COLOR = {'masked': (0, 200, 0), 'unmasked': (0, 0, 220), 'uncertain': (180, 130, 0)}
UNKNOWN_COLOR = (120, 120, 120)
IDENTITY_COLOR = (255, 255, 255)

def annotate(bgr_image, boxes, mask_labels, identities):
    img = bgr_image.copy()
    if boxes is None:
        return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    for (i, box) in enumerate(boxes):
        (x1, y1, x2, y2) = [int(c) for c in box]
        (mask_label, mask_conf) = mask_labels[i] if i < len(mask_labels) else ('?', 0.0)
        (identity, id_sim) = identities[i] if i < len(identities) else ('Unknown', 0.0)
        box_color = LABEL_COLOR.get(mask_label, UNKNOWN_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        mask_text = f'{mask_label} ({mask_conf:.0%})'
        ((tw, th), _) = cv2.getTextSize(mask_text, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(img, mask_text, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        id_text = f'{identity} ({id_sim:.2f})'
        cv2.putText(img, id_text, (x1, y2 + th + 6), font, font_scale, IDENTITY_COLOR, 1, cv2.LINE_AA)
    return img

def run_pipeline(image_path, config_path, output_path, show=False):
    print('\n' + '=' * 60)
    print('  Face AI Pipeline  --  Stage 1 -> Stage 2 -> Stage 3')
    print('=' * 60 + '\n')
    print(f'Input image: {image_path}')
    (pil_image, bgr_image) = load_image(image_path)
    print(f'Dimensions : {pil_image.width} x {pil_image.height} px\n')
    print('Stage 1: Face Detection')
    try:
        from src.detection.yolo_detector import YOLOFaceDetector
        detector = YOLOFaceDetector(config_path)
        (boxes, confs, face_crops) = detector.detect_and_crop(pil_image)
    except Exception as e:
        print(f'YOLO Error: {e}')
        print('No faces found.')
        save_image(bgr_image, output_path)
        return
    print(f'Faces detected and cropped: {len(face_crops)}\n')
    print('Stage 2: Mask Classification')
    classifier = MaskClassifier(config_path=config_path)
    mask_results = classifier.predict_batch(face_crops)
    for (idx, (label, conf)) in enumerate(mask_results):
        print(f'Face {idx + 1}: {label} ({conf:.1%} conf)')
    print()
    print('Stage 3: Face Recognition')
    recognizer = FaceRecognizer(config_path=config_path)
    if not recognizer.gallery:
        print('Gallery empty. Skipping recognition.')
        identity_results = [('Unknown', 0.0)] * len(face_crops)
    else:
        identity_results = recognizer.recognize_batch(face_crops)
        for (idx, (name, sim)) in enumerate(identity_results):
            print(f'Face {idx + 1}: {name} (sim = {sim:.4f})')
    print()
    annotated = annotate(bgr_image, boxes, mask_results, identity_results)
    save_image(annotated, output_path)
    print(f'Saved annotated output to {output_path}')
    if show:
        cv2.imshow('Face AI Pipeline — Output', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('\n' + '=' * 60)
    print('  Pipeline complete!')
    print('=' * 60 + '\n')
    return annotated

def parse_args():
    parser = argparse.ArgumentParser(description='Face AI Pipeline — end-to-end runner')
    parser.add_argument('--image', default='data/raw/test_image.jpg', help='Path to input image')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config YAML')
    parser.add_argument('--output', default='outputs/results/pipeline_output.jpg', help='Path to save annotated result')
    parser.add_argument('--show', action='store_true', help='Show the annotated image in a window')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.image):
        print(f"Downloading test face to {args.image}")
        import urllib.request
        os.makedirs(os.path.dirname(args.image), exist_ok=True)
        url = 'https://randomuser.me/api/portraits/men/75.jpg'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            with open(args.image, 'wb') as f:
                f.write(r.read())
    run_pipeline(args.image, args.config, args.output, show=args.show)
