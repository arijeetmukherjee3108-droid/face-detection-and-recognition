import os
import pickle
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
GALLERY_PATH = 'models/face_recognizer/gallery.pkl'

class FaceRecognizer:
    TRANSFORM = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __init__(self, config_path='configs/config.yaml'):
        config = load_config(config_path)
        rec_cfg = config['recognition']
        self.similarity_threshold = rec_cfg['similarity_threshold']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.gallery = {}
        self._load_gallery()

    def _load_gallery(self):
        if os.path.exists(GALLERY_PATH):
            with open(GALLERY_PATH, 'rb') as f:
                raw = pickle.load(f)
            self.gallery = {name: np.mean(embeddings, axis=0) for (name, embeddings) in raw.items()}
            print(f'Loaded {len(self.gallery)} identities from gallery.')
        else:
            print(f"Gallery not found at {GALLERY_PATH}.")

    def save_gallery(self, raw_gallery: dict):
        os.makedirs(os.path.dirname(GALLERY_PATH), exist_ok=True)
        with open(GALLERY_PATH, 'wb') as f:
            pickle.dump(raw_gallery, f)
        self.gallery = {name: np.mean(embs, axis=0) for (name, embs) in raw_gallery.items()}
        print(f'Saved gallery to {GALLERY_PATH}')

    @torch.no_grad()
    def embed(self, pil_image) -> np.ndarray:
        tensor = self.TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).squeeze(0).cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-08)
        return embedding

    @torch.no_grad()
    def embed_batch(self, pil_images) -> np.ndarray:
        tensors = torch.stack([self.TRANSFORM(img) for img in pil_images]).to(self.device)
        embeddings = self.model(tensors).cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-08
        return embeddings / norms

    def recognize(self, pil_image):
        if not self.gallery:
            return ('Unknown (no gallery)', 0.0)
        query_emb = self.embed(pil_image)
        best_name = 'Unknown'
        best_sim = -1.0
        for (name, gallery_emb) in self.gallery.items():
            sim = float(np.dot(query_emb, gallery_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim < self.similarity_threshold:
            return ('Unknown', best_sim)
        return (best_name, best_sim)

    def recognize_batch(self, pil_images):
        if not pil_images:
            return []
        embeddings = self.embed_batch(pil_images)
        results = []
        for emb in embeddings:
            if not self.gallery:
                results.append(('Unknown (no gallery)', 0.0))
                continue
            best_name = 'Unknown'
            best_sim = -1.0
            for (name, gallery_emb) in self.gallery.items():
                sim = float(np.dot(emb, gallery_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            if best_sim < self.similarity_threshold:
                best_name = 'Unknown'
            results.append((best_name, best_sim))
        return results
