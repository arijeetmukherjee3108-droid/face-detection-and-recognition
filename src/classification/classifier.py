import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_mobilenetv2(num_classes=2, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    return model

class MaskClassifier:
    TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __init__(self, config_path='configs/config.yaml'):
        config = load_config(config_path)
        cls_cfg = config['classification']
        self.classes = cls_cfg['classes']
        self.confidence_threshold = cls_cfg['confidence_threshold']
        self.model_path = cls_cfg['model_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        self.model = build_mobilenetv2(num_classes=len(self.classes), pretrained=True)
        if os.path.exists(self.model_path):
            print(f'Loading weights from {self.model_path}')
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            print('Weights not found. Using pretrained backbone.')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, pil_image):
        tensor = self.TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        (confidence, class_idx) = probs.max(0)
        confidence = confidence.item()
        label = self.classes[class_idx.item()]
        if confidence < self.confidence_threshold:
            label = 'uncertain'
        return (label, confidence)

    def predict_batch(self, pil_images):
        if not pil_images:
            return []
        tensors = torch.stack([self.TRANSFORM(img) for img in pil_images]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors)
            probs = torch.softmax(logits, dim=1)
        results = []
        for prob_row in probs:
            (confidence, class_idx) = prob_row.max(0)
            confidence = confidence.item()
            label = self.classes[class_idx.item()]
            if confidence < self.confidence_threshold:
                label = 'uncertain'
            results.append((label, confidence))
        return results
