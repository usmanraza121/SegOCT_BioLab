import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torch.amp import GradScaler, autocast
import numpy as np
import os
import time
import argparse
from sklearn.model_selection import KFold
from lib.dataloader import OCTSegDataset
from lib.metrics import compute_iou_and_miou, tcolormap
import logging
import datetime

# -----------------------------
# 1. Logging and Run Folder
# -----------------------------
def setup_run_dir(base_dir, fold=None, name=None):
    """Create a unique run directory for this training session, optionally with fold info."""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if fold is not None:
        run_dir = os.path.join(base_dir, f"{name}_{current_time}_fold_{fold}")
    else:
        run_dir = os.path.join(base_dir, f"run_{current_time}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")
    return run_dir, checkpoint_dir, log_file

# -----------------------------
# 2. Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 model for OCT segmentation with k-fold cross-validation")
    parser.add_argument('--name', type=str, default="deeplabv3_resnet50", help="Model name for saving checkpoints")
    parser.add_argument('--data_root', type=str, default="/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes", help="Dataset path")
    parser.add_argument('--num_classes', type=int, default=4, help="Number of classes")
    parser.add_argument('--crop_size', type=int, default=512, help="Crop size for images")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")
    parser.add_argument('--val_batch_size', type=int, default=4, help="Validation batch size")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--log_dir', type=str, default="runs", help="Base directory for run logs")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument('--early_stop_patience', type=int, default=5, help="Epochs to wait before early stopping")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds for cross-validation")
    return parser.parse_args()

# -----------------------------
# 3. Logger Setup
# -----------------------------
def setup_logger(log_file_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_file_path)])
    return logging.getLogger(__name__)

# -----------------------------
# 4. Model
# -----------------------------
def initialize_model(num_classes, device, resume_path=None):
    try:
        # Load pretrained DeepLabV3
        model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1,1))
        model = model.to(device)
        
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=False)
            logger.info(f"Loaded checkpoint from {resume_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

# -----------------------------
# 5. DataLoaders with K-Fold
# -----------------------------
def initialize_dataloaders(data_root, crop_size, batch_size, val_batch_size, fold, k_folds):
    dataset = OCTSegDataset(root_dir=data_root, split="train", crop_size=crop_size)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Get train and validation indices for the current fold
    indices = list(range(len(dataset)))
    train_idx, val_idx = list(kfold.split(indices))[fold]
    
    # Create subsets for training and validation
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Fold {fold+1}/{k_folds} - Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    return train_loader, val_loader

# -----------------------------
# 6. Training & Validation
# -----------------------------
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            out_dict = model(images)
            outputs = out_dict['out'] if isinstance(out_dict, dict) else out_dict
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / ((batch_idx+1) * train_loader.batch_size)
        logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_time = time.time() - start_time
    writer.add_scalar("Loss/Train", epoch_loss, epoch+1)
    logger.info(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Time: {epoch_time:.1f}s")
    return epoch_loss

def validate_epoch(model, val_loader, criterion, device, num_classes, epoch, writer):
    model.eval()
    val_loss = 0.0
    val_miou = 0.0
    val_ious = np.zeros(num_classes)
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            out_dict = model(images)
            outputs = out_dict['out'] if isinstance(out_dict, dict) else out_dict
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * images.size(0)
            miou, ious = compute_iou_and_miou(outputs, targets, num_classes=num_classes)
            val_miou += miou if not np.isnan(miou) else 0.0
            val_ious += np.array([iou if not np.isnan(iou) else 0.0 for iou in ious])
            num_batches += 1
            
            # Log sample images only once
            if epoch % 7 == 0 and num_batches == 10:
                sample_images = images[:4].cpu()
                sample_preds = torch.argmax(outputs[:4], dim=1).unsqueeze(1).cpu()
                color_preds = tcolormap(sample_preds, num_classes)
                sample_labels = targets[:4].unsqueeze(1).cpu()
                
                writer.add_images("Val/Images", sample_images, epoch+1)
                writer.add_images("Val/Predictions", sample_preds.float()/num_classes, epoch+1)
                writer.add_images("Val/ColorPredictions", color_preds, epoch+1)
                writer.add_images("Val/Labels", sample_labels.float()/num_classes, epoch+1)
    
    val_loss /= len(val_loader.dataset)
    val_miou /= num_batches
    val_ious /= num_batches
    
    writer.add_scalar("Loss/Validation", val_loss, epoch+1)
    writer.add_scalar("mIoU/Validation", val_miou, epoch+1)
    for i, iou in enumerate(val_ious):
        writer.add_scalar(f"IoU/Class_{i}", iou, epoch+1)
    
    logger.info(f"Validation | Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, IoUs: {val_ious}")
    return val_loss, val_miou, val_ious

# -----------------------------
# 7. Main Function with K-Fold
# -----------------------------
def main():
    global logger
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize lists to store metrics across folds
    fold_results = []
    
    for fold in range(args.k_folds):
        # Create run directory for this fold
        run_dir, checkpoint_dir, log_file = setup_run_dir(args.log_dir, fold=fold, name=args.name)
        logger = setup_logger(log_file)
        logger.info(f"Starting fold {fold+1}/{args.k_folds}")
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Using device: {device}")
        
        # Initialize model, dataloaders, loss, optimizer
        model = initialize_model(args.num_classes, device, args.resume)
        model.name = args.name
        train_loader, val_loader = initialize_dataloaders(args.data_root, args.crop_size, args.batch_size, args.val_batch_size, fold, args.k_folds)
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scaler = GradScaler()
        writer = SummaryWriter(log_dir=run_dir)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        fold_metrics = {'val_loss': [], 'val_miou': [], 'val_ious': []}
        
        for epoch in range(args.num_epochs):
            train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer)
            val_loss, val_miou, val_ious = validate_epoch(model, val_loader, criterion, device, args.num_classes, epoch, writer)
            
            fold_metrics['val_loss'].append(val_loss)
            fold_metrics['val_miou'].append(val_miou)
            fold_metrics['val_ious'].append(val_ious)
            
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model.name}_{fold+1}_epoch_{epoch+1}.pth")
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)
                logger.info(f"Saved periodic checkpoint at epoch {epoch+1} to {checkpoint_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, f"{model.name}_{fold+1}_best.pth")
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch+1}, checkpoint_path)
                logger.info(f"Saved best model at epoch {epoch+1} to {checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    logger.info(f"Early stopping triggered after {args.early_stop_patience} epochs with no improvement.")
                    break
            
            scheduler.step()
        
        # Store fold results
        fold_results.append(fold_metrics)
        writer.close()
        
        # Log fold summary
        avg_val_loss = np.mean(fold_metrics['val_loss'])
        std_val_loss = np.std(fold_metrics['val_loss'])
        avg_val_miou = np.mean(fold_metrics['val_miou'])
        std_val_miou = np.std(fold_metrics['val_miou'])
        avg_val_ious = np.mean(fold_metrics['val_ious'], axis=0)
        
        logger.info(f"Fold {fold+1} completed")
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        logger.info(f"Average Validation mIoU: {avg_val_miou:.4f} ± {std_val_miou:.4f}")
        logger.info(f"Average Per-class IoU: {', '.join([f'Class {i}: {iou:.4f}' for i, iou in enumerate(avg_val_ious)])}")
    
    # Compute and log cross-validation results
    all_val_losses = [np.mean(fold['val_loss']) for fold in fold_results]
    all_val_mious = [np.mean(fold['val_miou']) for fold in fold_results]
    all_val_ious = [np.mean(fold['val_ious'], axis=0) for fold in fold_results]
    
    cv_val_loss = np.mean(all_val_losses)
    cv_val_loss_std = np.std(all_val_losses)
    cv_val_miou = np.mean(all_val_mious)
    cv_val_miou_std = np.std(all_val_mious)
    cv_val_ious = np.mean(all_val_ious, axis=0)
    
    logger.info("Cross-validation completed")
    logger.info(f"Cross-validation Validation Loss: {cv_val_loss:.4f} ± {cv_val_loss_std:.4f}")
    logger.info(f"Cross-validation Validation mIoU: {cv_val_miou:.4f} ± {cv_val_miou_std:.4f}")
    logger.info(f"Cross-validation Per-class IoU: {', '.join([f'Class {i}: {iou:.4f}' for i, iou in enumerate(cv_val_ious)])}")

if __name__ == "__main__":
    main()