import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.utils import make_grid

from lib.dataloader import OCTSegDataset
from lib.metrics import compute_iou_and_miou, tcolormap  # assumes tcolormap -> (N,3,H,W) in [0,1]

# -----------------------------
# Reproducibility & performance
# -----------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # allow autotune for speed
    torch.backends.cudnn.benchmark = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()  # enable automatic mixed precision on GPU

# -----------------------------
# 1) Datasets & loaders
# -----------------------------
data_root = "/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes"
num_classes = 4

train_dataset = OCTSegDataset(root_dir=data_root, split="train")
val_dataset   = OCTSegDataset(root_dir=data_root, split="val")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# Quick sanity prints
img, label = train_dataset[0]
print("Train Image:", img.shape, "Train Label:", label.shape)
print("Unique classes in first label:", torch.unique(label))
print("#Train:", len(train_dataset), "| #Val:", len(val_dataset))

# -----------------------------
# 2) Model
# -----------------------------
# If using torchvision<=0.12 with PyTorch 1.11, pretrained=True is fine.
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
model.to(device)

# -----------------------------
# 3) Loss, optimizer, (optional) scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

scaler = torch.GradScaler('enabled' if use_amp else False)

# -----------------------------
# 4) TensorBoard
# -----------------------------
writer = SummaryWriter(log_dir='experiments/runs')

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Best-effort denorm for logging if inputs were normalized with ImageNet stats.
    x: (N,C,H,W) in torch.float32
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return torch.clamp(x * std + mean, 0, 1)

# -----------------------------
# 5) Train / Validate loops
# -----------------------------
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    seen = 0

    for batch_idx, (images, targets) in enumerate(train_loader, start=1):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(enabled=use_amp):
            out_dict = model(images)
            logits = out_dict['out'] if isinstance(out_dict, dict) else out_dict
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bsz = images.size(0)
        seen += bsz
        running_loss += loss.item() * bsz

        # Optional: batch-level logging (commented to keep TB light)
        # global_step = epoch * len(train_loader) + (batch_idx - 1)
        # writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        if batch_idx % 2 == 0 or batch_idx == len(train_loader):
            avg_so_far = running_loss / max(seen, 1)
            print(f"Epoch {epoch+1:02d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss.item():.4f} | Avg: {avg_so_far:.4f}")

    epoch_loss = running_loss / max(seen, 1)
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{num_epochs}]  TrainLoss: {epoch_loss:.4f}  Time: {epoch_time:.1f}s")
    writer.add_scalar('Train/Loss', epoch_loss, epoch+1)
    writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch+1)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_running = 0.0
    val_miou_sum = 0.0
    val_ious_sum = np.zeros(num_classes, dtype=np.float64)
    num_batches = 0

    last_images = None
    last_targets = None
    last_logits = None

    with torch.no_grad():
        for images, targets in val_loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            out_dict = model(images)
            logits = out_dict['out'] if isinstance(out_dict, dict) else out_dict
            loss = criterion(logits, targets)

            bsz = images.size(0)
            val_running += loss.item() * bsz

            # compute IoU/mIoU from logits
            miou, ious = compute_iou_and_miou(logits, targets, num_classes=num_classes)
            val_miou_sum += miou
            val_ious_sum += np.array([iou if not np.isnan(iou) else 0.0 for iou in ious], dtype=np.float64)
            num_batches += 1

            # keep last batch for logging
            last_images, last_targets, last_logits = images, targets, logits

    val_loss = val_running / max(len(val_loader.dataset), 1)
    val_miou = val_miou_sum / max(num_batches, 1)
    val_ious = (val_ious_sum / max(num_batches, 1)).astype(np.float32)

    results = {
        'val_loss': val_loss,
        'val_miou': val_miou,
        'val_ious': val_ious,
    }

    print(f"Validation  Loss: {val_loss:.4f}  mIoU: {val_miou:.4f}  IoUs: {val_ious:.4f}")
    # print('Per-class IoU: ' + ', '.join([f"Class {i}: {iou:.4f}" for i, iou in enumerate(val_ious)]))

    writer.add_scalar('Val/Loss', val_loss, epoch+1)
    writer.add_scalar('Val/mIoU', val_miou, epoch+1)
    for i, iou in enumerate(val_ious):
        writer.add_scalar(f'Val/IoU/Class_{i}', float(iou), epoch+1)

    # step scheduler on validation loss
    scheduler.step(val_loss)

    # -----------------------------
    # TensorBoard image logging (last val batch)
    # -----------------------------
    if last_images is not None:
        # If your inputs are ImageNet-normalized, denorm for visualization
        vis_images = denorm_imagenet(last_images.float()).cpu()

        # Predictions
        pred_classes = torch.argmax(last_logits, dim=1).unsqueeze(1)  # (N,1,H,W)
        color_preds = tcolormap(pred_classes, num_classes)  # (N,3,H,W) in [0,1]

        # Labels
        labels_1c = last_targets.unsqueeze(1)  # (N,1,H,W)
        # grayscale label preview (optional)
        labels_gray = labels_1c.float().cpu() / max(num_classes-1, 1)
        # color labels if you implement tcolormap for labels too
        color_labels = tcolormap(labels_1c, num_classes)

        # Make concise grids
        n_show = min(4, vis_images.size(0))
        grid_img   = make_grid(vis_images[:n_show], nrow=n_show)
        grid_predC = make_grid(color_preds[:n_show].cpu(), nrow=n_show)
        grid_labC  = make_grid(color_labels[:n_show].cpu(), nrow=n_show)

        writer.add_image('Val/Samples/Images', grid_img, epoch+1)
        writer.add_image('Val/Samples/PredictionsColor', grid_predC, epoch+1)
        writer.add_image('Val/Samples/LabelsColor', grid_labC, epoch+1)

    # -----------------------------
    # Checkpointing (save full state so you can resume)
    # -----------------------------
    ckpt_dir = 'experiments/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    # save best by val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_miou': val_miou,
        }, os.path.join(ckpt_dir, 'deeplabv3_best.pth'))
        print(f"Saved new best model at epoch {epoch+1}")

    # optional: always save last
    torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_miou': val_miou,
    }, os.path.join(ckpt_dir, 'deeplabv3_last.pth'))

# -----------------------------
# Done
# -----------------------------
writer.close()
print("\nTraining complete.")
