import urllib.request
import urllib.error
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.image_utils import load_image, draw_boxes, save_image
from src.detection.detector import FaceDetector

def download_test_image(save_path='data/raw/test_image.jpg'):
    urls = ['https://randomuser.me/api/portraits/men/75.jpg', 'https://randomuser.me/api/portraits/women/44.jpg']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        from PIL import Image as _PIL
        try:
            _PIL.open(save_path).verify()
            print(f'Test image already exists at {save_path} — skipping download.')
            return save_path
        except Exception:
            print('Existing file is corrupt — deleting and re-downloading...')
            os.remove(save_path)
    for url in urls:
        try:
            print(f'Downloading test image → {save_path} ...')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(save_path, 'wb') as f:
                    f.write(response.read())
            print('Download complete.')
            return save_path
        except Exception as e:
            print(f'  Failed ({url}): {e} — trying next URL...')
    raise RuntimeError('All download URLs failed. Please manually place a face photo at data/raw/test_image.jpg')

def run_test():
    print('\n' + '=' * 55)
    print('  Stage 1 — Face Detection Test')
    print('=' * 55 + '\n')
    image_path = download_test_image()
    (pil_image, bgr_image) = load_image(image_path)
    print(f'Image loaded: {pil_image.size[0]}×{pil_image.size[1]} px')
    detector = FaceDetector(config_path='configs/config.yaml')
    (boxes, landmarks) = detector.detect(pil_image)
    if boxes is None:
        print('\nNo faces detected. Try using a different test image.')
        print('   Replace the URL in download_test_image() with a clear face photo.')
        return
    annotated = draw_boxes(bgr_image, boxes, landmarks)
    save_image(annotated, 'outputs/results/test_detected.jpg')
    face_crops = detector.crop_faces(pil_image, boxes)
    os.makedirs('outputs/results/faces', exist_ok=True)
    for (idx, face) in enumerate(face_crops):
        face_path = f'outputs/results/faces/face_{idx:02d}.jpg'
        face.save(face_path)
        print(f'  Saved face crop: {face_path}')
    print('\n  Test complete!')
    print(f'   Annotated image -> outputs/results/test_detected.jpg')
    print(f'   Face crops      -> outputs/results/faces/')
    print('=' * 55 + '\n')
if __name__ == '__main__':
    run_test()
