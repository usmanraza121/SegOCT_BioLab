import os
import time
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.utils import make_grid

from PIL import Image

# Patch ANTIALIAS for compatibility with Pillow >= 10
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
# -----------------------------
# Import your modules
# -----------------------------
from lib.dataloader import OCTSegDataset
from lib.metrics import compute_iou_and_miou, tcolormap

# -----------------------------
# Device and AMP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

# -----------------------------
# Hyperparameters
# -----------------------------
num_classes = 4
batch_size_train = 8
batch_size_val = 4
learning_rate = 1e-4
num_epochs = 1
crop_size = 512
data_root = "/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes"

# -----------------------------
# Create run folder and logging
# -----------------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
run_name = f"DeepLabv3_Cityscapes_bs{batch_size_train}_lr{learning_rate}_{timestamp}"
log_dir = os.path.join("experiments", "runs", run_name)
os.makedirs(log_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# Logging
log_file = os.path.join(log_dir, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logging.info(f"TensorBoard & logs are saved in: {log_dir}")

# -----------------------------
# Datasets & DataLoaders
# -----------------------------
train_dataset = OCTSegDataset(root_dir=data_root, split="train")
val_dataset = OCTSegDataset(root_dir=data_root, split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False,
                        num_workers=4, pin_memory=True)

logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
logging.info(f"Unique classes in label: {torch.unique(train_dataset[0][1])}")

# -----------------------------
# Model
# -----------------------------
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1,1))
model.to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -----------------------------
# Training loop
# -----------------------------
best_val_loss = float('inf')
results = {'val_loss': [], 'val_miou': [], 'val_ious': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        avg_loss_so_far = running_loss / ((batch_idx + 1) * images.size(0))
        logging.info(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss_so_far:.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_time = time.time() - start_time
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.1f}s | Loss: {epoch_loss:.4f}")
    writer.add_scalar('Train/Loss', epoch_loss, epoch+1)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0.0
    val_miou = 0.0
    val_ious = np.zeros(num_classes)
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                loss = criterion(outputs, targets)
            
            val_loss += loss.item() * images.size(0)
            miou, ious = compute_iou_and_miou(outputs, targets, num_classes=num_classes)
            val_miou += miou
            val_ious += np.array([iou if not np.isnan(iou) else 0.0 for iou in ious])
            num_batches += 1

        val_loss /= len(val_loader.dataset)
        val_miou /= num_batches
        val_ious /= num_batches

    results['val_loss'].append(val_loss)
    results['val_miou'].append(val_miou)
    results['val_ious'].append(val_ious)

    logging.info(f"Validation Loss: {val_loss:.4f} | mIoU: {val_miou:.4f} | Per-class IoU: {val_ious}")

    writer.add_scalar('Val/Loss', val_loss, epoch+1)
    writer.add_scalar('Val/mIoU', val_miou, epoch+1)
    for i, iou in enumerate(val_ious):
        writer.add_scalar(f'Val/IoU/Class_{i}', iou, epoch+1)

    # -----------------------------
    # Sample images for TensorBoard
    # -----------------------------
    sample_images = images[:4].cpu()
    sample_preds = torch.argmax(outputs[:4], dim=1).unsqueeze(1).cpu()
    color_preds = tcolormap(sample_preds, num_classes)
    sample_labels = targets[:4].cpu()

    writer.add_images('Val/Images', sample_images, epoch+1)
    writer.add_images('Val/Predictions', sample_preds.float()/num_classes, epoch+1)
    writer.add_images('Val/ColorPredictions', color_preds, epoch+1)
    writer.add_images('Val/Labels', sample_labels.unsqueeze(1).float()/num_classes, epoch+1)

    # -----------------------------
    # Save best model
    # -----------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "deeplabv3_best.pth"))
        logging.info(f"Saved new best model at epoch {epoch+1}")

# -----------------------------
# Final summary
# -----------------------------
avg_val_loss = np.mean(results['val_loss'])
std_val_loss = np.std(results['val_loss'])
avg_val_miou = np.mean(results['val_miou'])
std_val_miou = np.std(results['val_miou'])
avg_val_ious = np.mean(results['val_ious'], axis=0)

logging.info(f"\nFinal Results:")
logging.info(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
logging.info(f"Average Validation mIoU: {avg_val_miou:.4f} ± {std_val_miou:.4f}")
logging.info(f"Average Per-class IoU: {', '.join([f'Class {i}: {iou:.4f}' for i, iou in enumerate(avg_val_ious)])}")

writer.close()
