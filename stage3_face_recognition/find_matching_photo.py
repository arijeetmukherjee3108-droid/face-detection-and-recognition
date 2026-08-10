import os
import sys
import pickle
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.yolo_detector import YOLOFaceDetector
from src.recognition.recognizer import FaceRecognizer

def build_or_load_db_embeddings(db_folder, recognizer, cache_path='models/face_recognizer/db_cache.pkl'):
    if os.path.exists(cache_path):
        print(f'[INFO] Loading database embeddings from cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    print(f'[INFO] Cache not found. Building embedding database for: {db_folder}')
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    db_embeddings = {}
    image_files = [f for f in os.listdir(db_folder) if os.path.splitext(f.lower())[1] in supported_exts]
    for img_file in tqdm(image_files, desc='Processing DB images'):
        img_path = os.path.join(db_folder, img_file)
        try:
            pil_img = Image.open(img_path).convert('RGB')
            emb = recognizer.embed(pil_img)
            db_embeddings[img_path] = emb
        except Exception as e:
            print(f'  [WARN] Skipping {img_file}: {e}')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(db_embeddings, f)
    print(f'[INFO] Saved {len(db_embeddings)} embeddings to cache.')
    return db_embeddings

def main():
    parser = argparse.ArgumentParser(description='Find matching photos based on a query face.')
    parser.add_argument('--query', required=True, help='Path to the query image')
    parser.add_argument('--db', required=True, help='Directory containing database images to search through')
    parser.add_argument('--threshold', type=float, default=0.6, help='Cosine similarity threshold (default: 0.6)')
    args = parser.parse_args()
    config_path = 'configs/config.yaml'
    print('\n' + '=' * 60)
    print('  Face AI - Find Matching Photo')
    print('=' * 60 + '\n')
    print('[INFO] Loading Detector and Recognizer...')
    detector = YOLOFaceDetector(config_path)
    recognizer = FaceRecognizer(config_path)
    db_embeddings = build_or_load_db_embeddings(args.db, recognizer)
    if not db_embeddings:
        print('[ERROR] No embeddings found in the database directory.')
        return
    print(f'\n[INFO] Processing query image: {args.query}')
    try:
        query_pil = Image.open(args.query).convert('RGB')
    except Exception as e:
        print(f'[ERROR] Could not load query image: {e}')
        return
    (boxes, confs, face_crops) = detector.detect_and_crop(query_pil)
    if not face_crops:
        print('[ERROR] No face detected in the query image. Try another photo.')
        face_crops = [query_pil.resize((160, 160))]
        print('[INFO] Falling back to using the entire query image as the face crop.')
    else:
        print(f'[INFO] Found {len(face_crops)} face(s) in query image. Using the first one.')
    query_crop = face_crops[0]
    query_emb = recognizer.embed(query_crop)
    print('\n[INFO] Searching database for a match...')
    best_match_path = None
    best_sim = -1.0
    for (db_path, db_emb) in db_embeddings.items():
        sim = float(np.dot(query_emb, db_emb))
        if sim > best_sim:
            best_sim = sim
            best_match_path = db_path
    if best_match_path and best_sim >= args.threshold:
        print(f'\n[SUCCESS] MATCH FOUND!')
        print(f'  --> Matching Photo: {best_match_path}')
        print(f'  --> Similarity Score: {best_sim:.4f}')
        q_cv2 = cv2.cvtColor(np.array(query_crop.resize((256, 256))), cv2.COLOR_RGB2BGR)
        db_pil = Image.open(best_match_path).convert('RGB')
        db_cv2 = cv2.cvtColor(np.array(db_pil.resize((256, 256))), cv2.COLOR_RGB2BGR)
        combined = np.hstack((q_cv2, db_cv2))
        cv2.putText(combined, 'Query Face', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f'Matched Photo ({best_sim:.2f})', (256 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Match Result', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f'\n[INFO] No strong match found.')
        if best_match_path:
            print(f'  --> Closest photo was: {best_match_path} (Score: {best_sim:.4f})')
            print(f'  --> Threshold required: {args.threshold}')
if __name__ == '__main__':
    main()
