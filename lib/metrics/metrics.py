from sympy import root
import torch

import os
from torchvision import models
import numpy as np

VOC_COLORMAP = np.array([
    [0, 0, 0],        # class 0 - background
    [128, 0, 0],      # class 1
    [0, 128, 0],      # class 2
    [128, 128, 0],    # class 3
    [0, 0, 128],      # class 4
    [128, 0, 128],    # class 5
    [0, 128, 128],    # class 6
    [128, 128, 128],  # class 7
    # add more colors if you have more classes
], dtype=np.uint8)

def decode_segmap(label_mask, num_classes):
    """Map class indices to RGB colors."""
    h, w = label_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        color_mask[label_mask == c] = VOC_COLORMAP[c % len(VOC_COLORMAP)]
    return color_mask

def tcolormap(tensor, num_classes):
    """Convert (N,1,H,W) tensor to RGB image tensor (N,3,H,W)."""
    tensor = tensor.squeeze(1).cpu().numpy()  # (N, H, W)
    color_images = [decode_segmap(mask, num_classes) for mask in tensor]
    color_images = np.stack(color_images)  # (N, H, W, 3)
    color_images = torch.from_numpy(color_images).permute(0, 3, 1, 2)  # (N,3,H,W)
    return color_images.float() / 255.0  # normalize to [0,1]

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


if __name__ == "__main__":
    tcolormap(torch.randint(0,4,(2,1,256,256)),4)
    compute_iou_and_miou(torch.randn(2,4,256,256), torch.randint(0,4,(2,256,256)), num_classes=4)
    print("Metrics module test passed.")
    print("tcolormap test passed.", tcolormap(torch.randint(0,4,(2,1,256,256)),4).shape)