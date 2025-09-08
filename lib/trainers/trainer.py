from sympy import root
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import OCTSegDataset
from deeplabv3_tnet.nets.model import tnet # replace with your model definition
import os
from torchvision import models
import numpy as np

# ---------------compute IoU and mIoU-------------
def compute_iou_and_miou(preds, labels, num_classes=4):
    # preds: (batch, num_classes, H, W), labels: (batch, H, W)
    preds = torch.argmax(preds, dim=1)  # (batch, H, W)
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    
    # Flatten predictions and labels
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    # Compute confusion matrix
    valid = (labels >= 0) & (labels < num_classes)  # Ignore invalid labels if any
    for p, t in zip(preds[valid], labels[valid]):
        confusion_matrix[p.long(), t.long()] += 1
    
    # Compute IoU per class
    ious = []
    for c in range(num_classes):
        TP = confusion_matrix[c, c]
        FP = confusion_matrix[c, :].sum() - TP
        FN = confusion_matrix[:, c].sum() - TP
        union = TP + FP + FN
        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = TP / (union + 1e-10)  # Avoid division by zero
        ious.append(iou.item())
    
    # Compute mIoU (ignore NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return miou, ious
# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Dataset
# -----------------------------
data_root = "/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes"  # change to your dataset folder
batch_size = 4
num_classes = 4
crop_size = 512

# train_dataset = OCTDataset(root=data_root, split='train', transform=get_transforms(crop_size))
# val_dataset = OCTDataset(root=data_root, split='val', transform=get_transforms(crop_size))

    # Train dataset
train_dataset = OCTSegDataset(root_dir=data_root, split="train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Validation dataset
val_dataset = OCTSegDataset(root_dir=data_root, split="val")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

img, label = train_dataset[0]
print("Train Image:", img.shape, "Train Label:", label.shape)
print("Unique classes in label:", torch.unique(label))
print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))


# -----------------------------
# 3. Model
# -----------------------------
# model = tnet.TNet(classes=num_classes)
# model = model.to(device)

# Load pretrained DeepLabV3
model = models.segmentation.deeplabv3_resnet50(pretrained=True)

# Replace classifier for 4 classes
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1,1))
model.to(device)

# # Optional: load checkpoint
# checkpoint_path = "checkpoints/latest_TNet_octdataset_os16.pth"
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['state_dict'] if "state_dict" in checkpoint else checkpoint)
#     print("Loaded checkpoint.")

# -----------------------------
# 4. Loss & optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Store results
results = {'val_loss': [], 'val_miou': [], 'val_ious': []}
# -----------------------------
# 5. Training loop
# -----------------------------
num_epochs = 10
best_val_loss = float('inf') 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # loss = criterion(outputs, targets)
        loss = criterion(outputs['out'], targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # -----------------------------
    # Optional: validation
    # -----------------------------
    model.eval()
    val_loss = 0.0
    val_miou = 0.0
    val_ious = np.zeros(num_classes)
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            # loss = criterion(outputs, targets)
            loss = criterion(outputs['out'], targets)
            val_loss += loss.item() * images.size(0)

            # Compute IoU and mIoU
            miou, ious = compute_iou_and_miou(outputs['out'], targets, num_classes=num_classes)
            val_miou += miou
            val_ious += np.array([iou if not np.isnan(iou) else 0.0 for iou in ious])
            num_batches += 1
            print(f"Validation mIoU: {miou:.4f}, IoUs: {ious}")

    val_loss /= len(val_loader.dataset)
    val_loss2 = val_loss / num_batches if num_batches > 0 else 0.0
    val_miou /= num_batches
    val_ious /= num_batches

    results['val_loss'].append(val_loss)
    results['val_miou'].append(val_miou)
    results['val_ious'].append(val_ious)

    print(f'Validation Loss: {val_loss:.4f}, Loss2: {val_loss2:.4f}')
    print(f"Validation mIoU: {val_miou:.4f}, IoUs: {val_ious}")
    print(f'Per-class IoU: {", ".join([f"Class {i}: {iou:.4f}" for i, iou in enumerate(val_ious)])}')
    # -----------------------------
    # Save best model
    # -----------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/deeplabv3_best.pth")
        print(f"Saved new best model at epoch {epoch+1}")

    # Save checkpoint
    # torch.save(model.state_dict(), f"checkpoints/TNet_epoch{epoch+1}.pth")
# Summarize results
avg_val_loss = np.mean(results['val_loss'])
std_val_loss = np.std(results['val_loss'])
avg_val_miou = np.mean(results['val_miou'])
std_val_miou = np.std(results['val_miou'])
avg_val_ious = np.mean(results['val_ious'], axis=0)

print(f'\nFinal Results')
print(f'Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}')
print(f'Average Validation mIoU: {avg_val_miou:.4f} ± {std_val_miou:.4f}')
print(f'Average Per-class IoU: {", ".join([f"Class {i}: {iou:.4f}" for i, iou in enumerate(avg_val_ious)])}')