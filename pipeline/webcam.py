import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from PIL import Image
from src.detection.yolo_detector import YOLOFaceDetector
from src.classification.classifier import MaskClassifier
from src.recognition.recognizer import FaceRecognizer
from pipeline.run_pipeline import annotate

def main():
    print('\n' + '=' * 60)
    print('  Face AI Pipeline  --  Real-time Webcam Inference')
    print('=' * 60 + '\n')
    config_path = 'configs/config.yaml'
    print('Loading models...')
    try:
        detector = YOLOFaceDetector(config_path)
    except Exception as e:
        print(f'Failed to load YOLO: {e}')
        return
    classifier = MaskClassifier(config_path=config_path)
    recognizer = FaceRecognizer(config_path=config_path)
    if not recognizer.gallery:
        print('Gallery empty. Recognition disabled.')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open webcam.')
        return
    print('\nStarting webcam...')
    print("Press 'q' to quit.\n")
    while True:
        (ret, frame) = cap.read()
        if not ret:
            print('Failed to grab frame.')
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        (boxes, confs, face_crops) = detector.detect_and_crop(pil_image)
        mask_results = []
        identity_results = []
        if len(face_crops) > 0:
            mask_results = classifier.predict_batch(face_crops)
            if recognizer.gallery:
                identity_results = recognizer.recognize_batch(face_crops)
            else:
                identity_results = [('Unknown', 0.0)] * len(face_crops)
        annotated_frame = annotate(frame, boxes, mask_results, identity_results)
        cv2.imshow('Face AI Pipeline - Live', annotated_frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('\nWebcam stream closed.')
if __name__ == '__main__':
    main()
