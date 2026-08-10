import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from src.classification.classifier import MaskClassifier
VAL_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def plot_confusion_matrix(cm, class_names, save_path):
    (fig, ax) = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names, title='Confusion Matrix — Mask Classifier', ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'Confusion matrix saved → {save_path}')

def main():
    DATA_DIR = 'data/face_crops'
    CONFIG = 'configs/config.yaml'
    CM_PATH = 'outputs/results/confusion_matrix.png'
    print('\n' + '=' * 55)
    print('  Stage 2 — Mask Classifier Evaluation')
    print('=' * 55 + '\n')
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=VAL_TRANSFORM)
    class_names = dataset.classes
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f'Evaluating on {len(dataset)} images | Classes: {class_names}')
    classifier = MaskClassifier(config_path=CONFIG)
    device = classifier.device
    all_preds = []
    all_labels = []
    classifier.model.eval()
    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            logits = classifier.model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    (precision, recall, f1, _) = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=list(range(len(class_names))))
    print(f'\nOverall Accuracy: {acc * 100:.2f}%\n')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, CM_PATH)
    print('\nEvaluation complete.')
if __name__ == '__main__':
    main()
