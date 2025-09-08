import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import models
from torch.amp import GradScaler, autocast
import numpy as np
import os
import time
import argparse
from lib.dataloader import OCTSegDataset
from lib.metrics import compute_iou_and_miou, tcolormap  # Ensure tcolormap is defined
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("experiments/runs/training.log")
                              ]
                    )
logger = logging.getLogger(__name__)

# Configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeepLabV3 model for semantic segmentation")
    parser.add_argument('--data_root', type=str, default="/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes", help="Path to dataset")
    parser.add_argument('--num_classes', type=int, default=4, help="Number of segmentation classes")
    parser.add_argument('--crop_size', type=int, default=512, help="Crop size for images")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--val_batch_size', type=int, default=4, help="Batch size for validation")
    parser.add_argument('--num_epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--checkpoint_dir', type=str, default="experiments/checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--log_dir', type=str, default="experiments/runs", help="TensorBoard log directory")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume training")
    return parser.parse_args()

def initialize_model(num_classes, device, resume_path=None):
    """Initialize the DeepLabV3 model and optionally load a checkpoint."""
    try:
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

def initialize_dataloaders(data_root, crop_size, batch_size, val_batch_size):
    """Initialize training and validation dataloaders."""
    try:
        train_dataset = OCTSegDataset(root_dir=data_root, split="train", crop_size=crop_size)
        val_dataset = OCTSegDataset(root_dir=data_root, split="val", crop_size=crop_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        return train_loader, val_loader
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with autocast(device_type=device):
            out_dict = model(images)
            outputs = out_dict['out'] if isinstance(out_dict, dict) else out_dict
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = running_loss / ((batch_idx + 1) * train_loader.batch_size)
            logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_time = time.time() - start_time
    writer.add_scalar('Train/Loss', epoch_loss, epoch+1)
    logger.info(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.1f}s")
    return epoch_loss

def validate_epoch(model, val_loader, criterion, device, num_classes, epoch, writer):
    """Run one validation epoch."""
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
            # logger.info(f"Validation | mIoU: {miou:.4f}, IoUs: {ious}")
            logger.info(f"Validation | mIoU: {miou:.4f}, IoUs: {', '.join([f'{iou:.4f}' for iou in ious])}")


            # Log sample images
            if num_batches == 1:
                sample_images = images[:4].cpu()
                sample_preds = torch.argmax(outputs[:4], dim=1).unsqueeze(1).cpu()
                color_preds = tcolormap(sample_preds, num_classes)
                sample_labels = targets[:4].cpu()
                
                writer.add_images('Val/Images', sample_images, epoch+1)
                writer.add_images('Val/Predictions', sample_preds.float()/num_classes, epoch+1)
                writer.add_images('Val/ColorPredictions', color_preds, epoch+1)
                writer.add_images('Val/Labels', sample_labels.unsqueeze(1).float()/num_classes, epoch+1)

    val_loss /= len(val_loader.dataset)
    val_miou /= num_batches
    val_ious /= num_batches
    
    writer.add_scalar('Val/Loss', val_loss, epoch+1)
    writer.add_scalar('Val/mIoU', val_miou, epoch+1)
    for i, iou in enumerate(val_ious):
        writer.add_scalar(f'Val/IoU/Class_{i}', iou, epoch+1)

    logger.info(f"Validation Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, IoUs: {val_ious:.4f}")
    # logger.info(f"Per-class IoU: {', '.join([f'Class {i}: {iou:.4f}' for i, iou in enumerate(val_ious)])}")
    return val_loss, val_miou, val_ious

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model, dataloaders, and other components
    model = initialize_model(args.num_classes, device, args.resume)
    train_loader, val_loader = initialize_dataloaders(args.data_root, args.crop_size, args.batch_size, args.val_batch_size)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler
    scaler = GradScaler()  # For mixed precision training
    writer = SummaryWriter(log_dir=args.log_dir)

    # Resume training from a checkpoint
    # if args.resume_path and os.path.exists(args.resume_path):
    #     checkpoint = torch.load(args.resume_path, map_location=device)
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     if 'optimizer' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #     if 'scaler' in checkpoint:
    #         scaler.load_state_dict(checkpoint['scaler'])
    #     start_epoch = checkpoint.get('epoch', 0)
    #     logger.info(f"Resuming training from epoch {start_epoch}")


    # Training loop
    best_val_loss = float('inf')
    results = {'val_loss': [], 'val_miou': [], 'val_ious': []}
    
    for epoch in range(args.num_epochs):
        train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer)
        val_loss, val_miou, val_ious = validate_epoch(model, val_loader, criterion, device, args.num_classes, epoch, writer)
        
        results['val_loss'].append(val_loss)
        results['val_miou'].append(val_miou)
        results['val_ious'].append(val_ious)
        
        # Save best model
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "deeplabv3_best.pth")
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch+1}, checkpoint_path)
            logger.info(f"Saved new best model at epoch {epoch+1} to {checkpoint_path}")
        # Always save last model
        torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch+1
                }, os.path.join(args.checkpoint_dir, f"deeplabv3_last.pth"))
        
        scheduler.step()  # Update learning rate
    
    # Summarize results
    avg_val_loss = np.mean(results['val_loss'])
    std_val_loss = np.std(results['val_loss'])
    avg_val_miou = np.mean(results['val_miou'])
    std_val_miou = np.std(results['val_miou'])
    avg_val_ious = np.mean(results['val_ious'], axis=0)
    
    logger.info('\nFinal Results')
    logger.info(f'Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}')
    logger.info(f'Average Validation mIoU: {avg_val_miou:.4f} ± {std_val_miou:.4f}')
    logger.info(f'Average Per-class IoU: {", ".join([f"Class {i}: {iou:.4f}" for i, iou in enumerate(avg_val_ious)])}')
    
    writer.close()

if __name__ == "__main__":
    main()