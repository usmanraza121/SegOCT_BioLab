import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4, lambda_dice=1.0, lambda_focal=1.0, alpha=0.25, gamma=2.0, class_weights=None):
        """
        num_classes : int : number of segmentation classes
        lambda_dice : float : weight for dice loss
        lambda_focal : float : weight for focal loss
        alpha, gamma : parameters for focal loss
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        """
        preds : [B, C, H, W] raw logits
        targets : [B, H, W] ground truth class indices
        """
        # ---- Cross Entropy Loss ----
        ce_loss = self.ce(preds, targets)

        # ---- Dice Loss ----
        preds_soft = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0,3,1,2).float()

        # intersection = (preds_soft * targets_onehot).sum(dim=(2,3))
        # union = preds_soft.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        # dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        # dice_loss = 1.0 - dice_score.mean()

        intersection = (preds_soft * targets_onehot).sum(dim=(0,2,3))  # sum over batch+H+W → [C]
        union = preds_soft.sum(dim=(0,2,3)) + targets_onehot.sum(dim=(0,2,3))
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice_score.mean()
        

        # dice_loss_per_class = 1.0 - dice_score
        # dice_weights = torch.tensor([1.0, 1.0, 1.0, 3.0]).to(preds.device)

        # dice_weights = dice_weights[:dice_loss_per_class.size(0)]
        # dice_loss = (dice_loss_per_class * dice_weights).sum() / dice_weights.sum()

        # ---- Focal Loss ----
        pt = (preds_soft * targets_onehot).sum(1)  # [B, H, W]
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * pt.log()
        focal_loss = focal_loss.mean()

        # ---- Combined Loss ----
        total_loss = ce_loss + self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss
# Example usage:
# criterion = CombinedLoss(num_classes=4, lambda_dice=1.0, lambda_focal=1.0)
# loss = criterion(predictions, targets)    
if __name__ == "__main__":
    # Simple test
    criterion = CombinedLoss(num_classes=4, lambda_dice=1.0, lambda_focal=1.0)
    preds = torch.randn(2, 4, 256, 256)  # Example predictions [B, C, H, W]
    targets = torch.randint(0, 4, (2, 256, 256))  # Example targets [B, H, W]
    loss = criterion(preds, targets)
    print("Loss:", loss.item())