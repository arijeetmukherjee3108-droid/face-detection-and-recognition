# 🔍 Face Detection and Recognition Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)]()

A modular, robust three-stage Artificial Intelligence pipeline designed for end-to-end face processing, built with PyTorch and OpenCV. This system seamlessly integrates face localization, safety compliance (mask classification), and biometric identification.

---

## 🌟 Key Features

*   **Stage 1 - Face Detection:** Accurately locates all faces within a given frame using state-of-the-art YOLOv8 or MTCNN architectures.
*   **Stage 2 - Mask Classification:** Evaluates each detected face to determine whether the individual is wearing a protective mask (using a custom PyTorch CNN).
*   **Stage 3 - Identity Recognition:** Identifies individuals against a pre-built gallery using FaceNet (PyTorch) embeddings and cosine similarity.
*   **Real-time Inference:** Fully optimized pipeline capable of processing live webcam streams with real-time on-screen annotations.

---

## 🏗️ Project Architecture

```text
face-detection-and-recognition/
├── data/
│   ├── annotations/       # Labels and metadata
│   ├── identity_gallery/  # Reference images for facial recognition
│   ├── raw/               # Original test images/videos
│   └── processed/         # Preprocessed dataset
├── src/
│   ├── detection/         # Face bounding box extraction modules
│   ├── classification/    # Mask/No-Mask classification modules
│   ├── recognition/       # Identity embedding and matching modules
│   └── utils/             # Shared utilities (image processing, etc.)
├── pipeline/              # Integration scripts (webcam.py, run_pipeline.py)
├── stage1_detection/      # YOLO/MTCNN training and testing scripts
├── stage2_mask_classification/ # CNN training scripts
├── stage3_face_recognition/ # Gallery building and recognition scripts
├── models/                # Saved model weights and compiled galleries
├── configs/               # YAML configuration parameters (config.yaml)
└── outputs/               # Training logs, resulting images, and crops
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/arijeetmukherjee3108-droid/face-detection-and-recognition.git
cd face-detection-and-recognition
```

### 2. Set up the Environment

It is highly recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
# source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Testing Procedures

To ensure that the pipeline components are functioning correctly, you can run isolated tests for the detection and recognition modules, as well as the integrated real-time webcam pipeline.

### Part 1: Testing Face Detection (Stage 1)

The detection test script downloads a sample test image, initializes the face detector, draws bounding boxes around detected faces, and saves cropped face images to the `outputs` directory.

1.  **Run the detection test script:**
    ```bash
    python test_detection.py
    ```
2.  **Verify Results:**
    *   Check `outputs/results/test_detected.jpg` to see the original image with bounding boxes drawn over detected faces.
    *   Check the `outputs/results/faces/` directory to see the individually cropped faces.

### Part 2: Testing Face Recognition (Stage 3)

The recognition test script requires a built gallery of known identities. It extracts face embeddings from a query image and compares them against the gallery to find the closest match.

1.  **Build the Identity Gallery:**
    Before testing recognition, ensure you have placed reference photos organized by name in `data/identity_gallery/<PersonName>/<photo.jpg>`. Then build the gallery pickle file:
    ```bash
    python stage3_face_recognition/build_gallery.py
    ```
2.  **Run the Recognition Test:**
    ```bash
    python stage3_face_recognition/test_recognition.py
    ```
3.  **Verify Results:**
    The console will output the detected identities and their respective cosine similarity scores against the gallery matches.

### Part 3: Full Real-Time Pipeline Test

To test the complete end-to-end integration (Detection -> Mask Classification -> Recognition) via your webcam:

1.  **Run the webcam pipeline script:**
    ```bash
    python pipeline/webcam.py
    ```
2.  **Interaction:**
    *   The webcam window will open, displaying real-time bounding boxes, mask status, and identity labels.
    *   Press the `q` key on your keyboard to safely terminate the video stream.

---

## 🛠️ Tech Stack

| Component           | Technology / Framework       |
|---------------------|------------------------------|
| **Face Detection**  | YOLOv8, MTCNN, OpenCV        |
| **Classification**  | PyTorch (Custom CNN)         |
| **Recognition**     | FaceNet (`facenet-pytorch`)  |
| **Data Processing** | NumPy, Pillow                |
| **Experiments**     | Jupyter Notebooks            |

---

## 👤 Author

**Arijeet Mukherjee**
*   GitHub: [@arijeetmukherjee3108-droid](https://github.com/arijeetmukherjee3108-droid)

---

## 📄 License

This project is intended for personal and educational use.
