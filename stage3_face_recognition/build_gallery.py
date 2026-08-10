import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from tqdm import tqdm
from src.recognition.recognizer import FaceRecognizer
GALLERY_DIR = 'data/identity_gallery'
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def build_gallery(gallery_dir=GALLERY_DIR):
    if not os.path.isdir(gallery_dir):
        print(f'Gallery directory not found: {gallery_dir}')
        print('Create identity-wise subfolders and add photos.')
        return
    identities = sorted([d for d in os.listdir(gallery_dir) if os.path.isdir(os.path.join(gallery_dir, d))])
    if not identities:
        print(f"No identity subfolders found in '{gallery_dir}'.")
        print('Expected: Person_A/, Person_B/, etc.')
        return
    print(f'\nFound {len(identities)} identities: {identities}\n')
    from src.detection.yolo_detector import YOLOFaceDetector
    detector = YOLOFaceDetector()
    recognizer = FaceRecognizer()
    raw_gallery = {}
    for person in tqdm(identities, desc='Building gallery'):
        person_dir = os.path.join(gallery_dir, person)
        image_files = [f for f in os.listdir(person_dir) if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS]
        if not image_files:
            print(f"No images found for '{person}'")
            continue
        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                pil_img = Image.open(img_path).convert('RGB')
                boxes, confs, face_crops = detector.detect_and_crop(pil_img)
                if not face_crops:
                    print(f'No face detected in {img_file}, skipping.')
                    continue
                emb = recognizer.embed(face_crops[0])
                embeddings.append(emb)
            except Exception as e:
                print(f'Skipping {img_file}: {e}')
        if embeddings:
            raw_gallery[person] = embeddings
            print(f'{person}: {len(embeddings)} embeddings')
        else:
            print(f'{person}: no valid embeddings')
    if raw_gallery:
        recognizer.save_gallery(raw_gallery)
        print(f'\nGallery built successfully with {len(raw_gallery)} identities.')
    else:
        print('\nNo embeddings generated.')
if __name__ == '__main__':
    print('\n' + '=' * 55)
    print('  Stage 3 — Face Recognition Gallery Builder')
    print('=' * 55)
    build_gallery()
    print('=' * 55 + '\n')
