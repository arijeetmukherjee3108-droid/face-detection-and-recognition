import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from src.classification.classifier import build_mobilenetv2

def load_config(path='configs/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
VAL_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_dataloaders(data_dir='data/face_crops', val_split=0.2, batch_size=16):
    full_dataset = datasets.ImageFolder(root=data_dir, transform=TRAIN_TRANSFORM)
    class_names = full_dataset.classes
    print(f'Classes found: {class_names}')
    print(f'Total images : {len(full_dataset)}')
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    (train_dataset, val_dataset) = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f'Train: {train_size} images  |  Val: {val_size} images')
    return (train_loader, val_loader, class_names)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for (images, labels) in tqdm(loader, desc='  Train', leave=False):
        (images, labels) = (images.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        (_, predicted) = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return (running_loss / total, 100.0 * correct / total)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in tqdm(loader, desc='  Val  ', leave=False):
            (images, labels) = (images.to(device), labels.to(device))
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            (_, predicted) = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return (running_loss / total, 100.0 * correct / total)

def main():
    config = load_config('configs/config.yaml')
    cls_cfg = config['classification']
    EPOCHS = 20
    LR = 0.001
    BATCH_SIZE = 16
    DATA_DIR = 'data/face_crops'
    SAVE_PATH = cls_cfg['model_path']
    FREEZE_EPOCHS = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    (train_loader, val_loader, class_names) = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    num_classes = len(class_names)
    model = build_mobilenetv2(num_classes=num_classes, pretrained=True)
    model.to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print(f'\nTraining for {EPOCHS} epochs (backbone frozen for first {FREEZE_EPOCHS})...')
    print('=' * 60)
    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            print(f'\n  [Epoch {epoch}] Unfreezing backbone — fine-tuning all layers')
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        (train_loss, train_acc) = train_one_epoch(model, train_loader, criterion, optimizer, device)
        (val_loss, val_acc) = validate(model, val_loader, criterion, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Epoch [{epoch:02d}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  [BEST] New best model saved ({val_acc:.1f}%) -> {SAVE_PATH}')
    print(f'\nTraining complete. Best val accuracy: {best_val_acc:.1f}%')
    plot_path = 'outputs/results/mask_classifier_training_curve.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, EPOCHS + 1)
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'], label='Val')
    ax2.set_title('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Training curve saved → {plot_path}')
if __name__ == '__main__':
    main()
