from sympy import root
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from lib.dataloader import OCTSegDataset
# from deeplabv3_tnet.nets.model import tnet # replace with your model definition
import os
from torchvision import models
import numpy as np
from lib.metrics import compute_iou_and_miou, tcolormap
from torchvision.utils import make_grid
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Dataset
# -----------------------------
data_root = "/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes"  # change to your dataset folder

num_classes = 4
crop_size = 512

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
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)

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

# TensorBoard writer
run_name = f"DeepLabv3_run_{time.strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=f"experiments/runs/{run_name}")

# Store results
results = {'val_loss': [], 'val_miou': [], 'val_ious': []}

# -----------------------------
# 5. Training loop
# -----------------------------
num_epochs = 2
best_val_loss = float('inf') 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        else:
            outputs = outputs

        loss = criterion(outputs, targets)
        # loss = criterion(outputs['out'], targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # Print every 1 batches
        if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss_so_far = running_loss / ((batch_idx + 1) * images.size(0))
                print(f"Batch {batch_idx + 1}/{len(train_loader)}    | Loss: {loss.item():.4f} | Avg Loss: {avg_loss_so_far:.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.1f}s")
    writer.add_scalar('Train/Loss', epoch_loss, epoch+1)

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
            if isinstance(outputs, dict):
                outputs = outputs['out']
            else:
                outputs = outputs
            loss = criterion(outputs, targets)
            # loss = criterion(outputs['out'], targets)
            val_loss += loss.item() * images.size(0)

            # Compute IoU and mIoU
            miou, ious = compute_iou_and_miou(outputs, targets, num_classes=num_classes)
            val_miou += miou
            val_ious += np.array([iou if not np.isnan(iou) else 0.0 for iou in ious])
            num_batches += 1
            print(f"Validation mIoU: {miou:.4f}, IoUs: {ious}")

    val_loss /= len(val_loader.dataset)
    val_miou /= num_batches
    val_ious /= num_batches

    results['val_loss'].append(val_loss)
    results['val_miou'].append(val_miou)
    results['val_ious'].append(val_ious)

    print(f'Validation Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, IoUs: {val_ious}')
    
    writer.add_scalar('Val/Loss', val_loss, epoch+1)
    writer.add_scalar('Val/mIoU', val_miou, epoch+1)
    for i, iou in enumerate(val_ious):
        writer.add_scalar(f'Val/IoU/Class_{i}', iou, epoch+1)

    sample_images = images[:4].cpu()
    sample_preds = torch.argmax(outputs[:4], dim=1).unsqueeze(1).cpu()
    color_preds = tcolormap(sample_preds, num_classes)
    sample_labels = targets[:4].cpu()

    writer.add_images('Val/Images', sample_images, epoch+1)
    # writer.add_images('Val/Predictions', sample_preds.float()/num_classes, epoch+1)
    writer.add_images('Val/ColorPredictions', color_preds, epoch+1)
    # writer.add_images('Val/Predictions', sample_preds, epoch)
    writer.add_images('Val/Labels', sample_labels.unsqueeze(1).float()/num_classes, epoch+1)

    # grid = make_grid(sample_images, nrow=4)
    # writer.add_image('Val/SampleImages', grid, epoch+1)
    # grid = make_grid(sample_preds, nrow=4)
    # writer.add_image('Val/SamplePredictions', grid, epoch+1)

    # -----------------------------
    # Save best model
    # -----------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("experiments/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "experiments/checkpoints/deeplabv3_best.pth")
        print(f"Saved new best model at epoch {epoch+1}")

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

writer.close()