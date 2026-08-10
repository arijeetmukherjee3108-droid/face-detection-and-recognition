import os
import sys
import shutil
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_YAML = 'configs/yolo_dataset.yaml'
MODEL_SIZE = 'yolov8n.pt'
EPOCHS = 30
BATCH_SIZE = 8
IMG_SIZE = 416
PROJECT_DIR = 'models/yolo'
RUN_NAME = 'face_detector'
DEVICE = 'cpu'

def check_dataset():
    with open(DATASET_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    root = cfg['path']
    train_dir = os.path.join(root, cfg['train'])
    val_dir = os.path.join(root, cfg['val'])
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f'Train images not found: {train_dir}\nMake sure you extracted the ZIP first:\n  D:/face-ai-pipeline/data/raw/face-detection-dataset/images/train/')
    train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.isdir(val_dir) else 0
    print(f'Dataset OK:')
    print(f'  Train: {train_count:,} images  ({train_dir})')
    print(f'  Val  : {val_count:,} images  ({val_dir})')
    return (train_count, val_count)

def main():
    print('\n' + '=' * 60)
    print('  Stage 1 -- YOLOv8 Face Detector Training')
    print('=' * 60 + '\n')
    (train_count, val_count) = check_dataset()
    try:
        from ultralytics import YOLO
    except ImportError:
        print('[ERROR] ultralytics not installed.')
        print('  Run: pip install ultralytics')
        sys.exit(1)
    print(f'\nModel      : {MODEL_SIZE}')
    print(f'Epochs     : {EPOCHS}')
    print(f'Batch size : {BATCH_SIZE}')
    print(f'Image size : {IMG_SIZE}')
    print(f'Device     : {DEVICE}')
    print(f'Output     : {PROJECT_DIR}/{RUN_NAME}/\n')
    model = YOLO(MODEL_SIZE)
    results = model.train(data=DATASET_YAML, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, project=PROJECT_DIR, name=RUN_NAME, device=DEVICE, patience=10, save=True, exist_ok=True, plots=True, verbose=True)
    best_src = os.path.join(PROJECT_DIR, RUN_NAME, 'weights', 'best.pt')
    best_dst = 'models/yolo/best.pt'
    os.makedirs(os.path.dirname(best_dst), exist_ok=True)
    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f'\n[OK] Best weights copied -> {best_dst}')
    results_src = os.path.join(PROJECT_DIR, RUN_NAME)
    results_dst = 'outputs/results/yolo_training'
    if os.path.isdir(results_src):
        if os.path.exists(results_dst):
            shutil.rmtree(results_dst)
        shutil.copytree(results_src, results_dst)
        print(f'[OK] Training results -> {results_dst}/')
    print('\n' + '=' * 60)
    print('  Training complete!')
    print(f'  Best model  -> {best_dst}')
    print(f'  Full run    -> {results_dst}/')
    print('=' * 60 + '\n')
if __name__ == '__main__':
    main()
