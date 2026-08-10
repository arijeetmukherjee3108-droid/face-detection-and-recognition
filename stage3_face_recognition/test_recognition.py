import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request
from PIL import Image
from src.recognition.recognizer import FaceRecognizer
from src.detection.detector import FaceDetector
from src.utils.image_utils import load_image

def main():
    print('\n' + '=' * 55)
    print('  Stage 3 — Face Recognition Test')
    print('=' * 55 + '\n')
    GALLERY_PATH = 'models/face_recognizer/gallery.pkl'
    if not os.path.exists(GALLERY_PATH):
        print('[INFO] No gallery found. Running gallery builder first...\n')
        from stage3_face_recognition.build_gallery import build_gallery
        build_gallery()
    recognizer = FaceRecognizer(config_path='configs/config.yaml')
    if not recognizer.gallery:
        print('\n[INFO] Gallery is empty. Add identity images under:\n       data/identity_gallery/<PersonName>/<photo.jpg>\nThen re-run this script.')
        return
    query_path = 'data/raw/test_image.jpg'
    if not os.path.exists(query_path):
        print(f'Downloading test image → {query_path}')
        os.makedirs('data/raw', exist_ok=True)
        url = 'https://randomuser.me/api/portraits/men/75.jpg'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(query_path, 'wb') as f:
                f.write(resp.read())
    (pil_image, _) = load_image(query_path)
    print(f'Query image loaded: {pil_image.size}')
    detector = FaceDetector(config_path='configs/config.yaml')
    (boxes, _) = detector.detect(pil_image)
    if boxes is None:
        print('[INFO] No high-confidence faces detected in the test image.')
        print('       Testing recognition directly on the full image as a fallback.')
        face_crops = [pil_image.resize((160, 160))]
    else:
        face_crops = detector.crop_faces(pil_image, boxes)
    print(f'\nRecognizing {len(face_crops)} face(s)...\n')
    for (idx, crop) in enumerate(face_crops):
        (identity, similarity) = recognizer.recognize(crop)
        print(f'  Face {idx + 1}: {identity}  (cosine similarity = {similarity:.4f})')
    print('\n' + '=' * 55)
    print('  Test complete!')
    print('=' * 55 + '\n')
if __name__ == '__main__':
    main()
