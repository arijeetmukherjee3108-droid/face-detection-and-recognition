# Face Recognition Module: Work Breakdown & Documentation

This document outlines the architecture of the Face Recognition module and provides a logical, 50/50 division of labor tailored to your team's skill sets. It also includes slide-ready notes for your upcoming presentation.

---

## 1. Architecture Overview

The face recognition stage of the pipeline consists of two main components:
1. **The Core Recognizer (`src/recognition/recognizer.py`)**: A class responsible for loading a pretrained deep learning model (FaceNet/InceptionResnetV1), transforming images into tensors, generating 512-dimensional embeddings, and comparing these embeddings using cosine similarity.
2. **The Gallery Builder (`stage3_face_recognition/build_gallery.py`)**: A data pipeline script that iterates through directories of known identities, processes the images through the recognizer, and saves the aggregated "average" embeddings into a serialized `.pkl` gallery.

---

## 2. Work Division

Since your teammate has less machine learning experience, the workload is divided so that **you handle the core Deep Learning and Tensor operations**, while **your teammate handles the Data Pipeline, File I/O, and Application Logic.**

### 🧑‍💻 Person A (You): Core ML Engine (Lead ML Developer)
**Focus:** PyTorch, Neural Networks, Linear Algebra, GPU Optimization.

*   **Model Initialization & Device Management:** 
    *   Loading the `InceptionResnetV1` (vggface2) model.
    *   Handling hardware acceleration (`cuda` vs `cpu`).
*   **Tensor Transformations:** 
    *   Designing the `torchvision.transforms` pipeline (Resizing to 160x160, normalization, tensor conversion).
*   **Embedding Generation (`embed` & `embed_batch`):** 
    *   Handling the forward pass through the network under `@torch.no_grad()`.
    *   L2 Normalization of the output vectors.
*   **Similarity Computation (`recognize` & `recognize_batch`):** 
    *   Implementing the mathematical logic for vector comparison (Dot product for Cosine Similarity).
    *   Applying the confidence threshold logic.

### 🧑‍💻 Person B (Teammate): Data Pipeline & Integration (Pipeline Developer)
**Focus:** Python Scripting, File I/O, Data Parsing, State Management.

*   **Gallery Building Script (`build_gallery.py`):** 
    *   Writing the logic to traverse the `data/identity_gallery` directory using `os`.
    *   Filtering supported image extensions (`.jpg`, `.png`, etc.).
    *   Adding CLI progress tracking using `tqdm`.
*   **Data I/O & Preprocessing:** 
    *   Loading images from disk and converting them to standard RGB using `PIL.Image`.
    *   Handling corrupted images or read errors gracefully via `try-except` blocks.
*   **State Management (`_load_gallery` & `save_gallery`):** 
    *   Serializing (saving) and deserializing (loading) the embeddings dictionary using Python's `pickle` library.
    *   Averaging multiple embeddings for a single person using `numpy.mean`.
*   **Configuration Management:** 
    *   Parsing YAML files (`load_config`) to dynamically set thresholds and paths without hardcoding.

---

## 3. Presentation Notes (Slide-by-Slide Guide)

You can use these bullet points directly for your presentation slides.

### 📝 Slide 1: Stage 3 Overview - Face Recognition
*   **Objective:** Identify detected faces by comparing them against a database of known individuals.
*   **Technology Stack:** PyTorch, FaceNet (InceptionResnetV1), Python, OpenCV/PIL.
*   **Workflow:** 
    1. Build a gallery of known faces.
    2. Extract facial features (embeddings) from new video frames.
    3. Compare new embeddings to the gallery to find the closest match.

### 📝 Slide 2: The Core ML Engine (Presented by You)
*   **The Model:** We utilize **FaceNet (InceptionResnetV1)**, pretrained on the VGGFace2 dataset, optimized for feature extraction.
*   **Feature Embeddings:** The model maps a face image into a 512-dimensional vector space where faces of the same person are mathematically closer together.
*   **Batch Processing:** Designed `embed_batch` to process multiple faces simultaneously on the GPU for real-time pipeline performance.
*   **Matching Algorithm:** We use **Cosine Similarity** (computed via dot products of L2-normalized vectors) to compare incoming faces with our gallery, applying a strict threshold to filter out "Unknown" faces.

### 📝 Slide 3: The Data & Gallery Pipeline (Presented by Teammate)
*   **Gallery Generation:** Developed an automated script (`build_gallery.py`) to ingest folders of identity images.
*   **Data Handling:** Uses `PIL` for robust image loading and standardizes all inputs to RGB before they hit the ML model.
*   **State Persistence:** The generated embeddings are aggregated (averaged) per identity and serialized using `pickle` for instant loading during live inference.
*   **Error Handling:** The pipeline skips corrupted files and logs warnings, ensuring the gallery building process doesn't crash on bad data.

### 📝 Slide 4: Challenges & Optimizations
*   **Challenge:** Variations in lighting and angles.
    *   *Solution (Teammate):* Averaged multiple embeddings per person during the gallery build phase to create a more robust "master" embedding.
*   **Challenge:** Real-time performance bottlenecks.
    *   *Solution (You):* Enforced `@torch.no_grad()` to prevent memory leaks and utilized batched tensor operations to maximize GPU utilization.
