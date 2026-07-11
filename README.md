# Face Detection and Recognition Pipeline

A modular three-stage AI pipeline for face detection, mask classification, and identity recognition.

---

## Pipeline Overview

Stage 1 --> Face Detection (locate all faces in frame)
Stage 2 --> Mask Classification (masked or unmasked)
Stage 3 --> Identity Recognition (who is this person)

---

## Project Structure

    face-detection-and-recognition/
    |-- data/
    |   |-- raw/               # Original images/videos (gitignored)
    |   |-- processed/         # Preprocessed data (gitignored)
    |   `-- annotations/       # Labels and metadata
    |-- notebooks/             # Jupyter notebooks for experiments
    |-- src/
    |   |-- detection/         # Face detection module
    |   |-- classification/    # Mask classification module
    |   |-- recognition/       # Identity recognition module
    |   `-- utils/             # Shared utilities
    |-- models/
    |   `-- weights/           # Trained model weights (gitignored)
    |-- configs/               # YAML configuration files
    |-- outputs/
    |   |-- logs/              # Training logs
    |   `-- results/           # Inference results
    |-- requirements.txt
    `-- README.md

---

## Getting Started

### 1. Clone the repository

    git clone https://github.com/arijeetmukherjee3108-droid/face-detection-and-recognition.git
    cd face-detection-and-recognition

### 2. Create a virtual environment

    python -m venv venv
    venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

---

## Tech Stack

| Component           | Technology              |
|---------------------|-------------------------|
| Face Detection      | OpenCV, MTCNN           |
| Mask Classification | PyTorch CNN             |
| Face Recognition    | FaceNet (facenet-pytorch)|
| Preprocessing       | NumPy, Pillow           |
| Experiments         | Jupyter Notebooks       |

---

## Roadmap

- [ ] Stage 1: Face Detection module
- [ ] Stage 2: Mask Classification model
- [ ] Stage 3: Face Recognition with FaceNet
- [ ] End-to-end pipeline integration
- [ ] Real-time webcam inference

---

## Author

Arijeet Mukherjee
GitHub: https://github.com/arijeetmukherjee3108-droid

---

## License

This project is for personal and educational use.
