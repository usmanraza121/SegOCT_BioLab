import os
from PIL import Image
from matplotlib import transforms
import numpy as np
from collections import namedtuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OCTSegDataset(Dataset):
    def __init__(self, root_dir, split="train", image_transform=None, crop_size=(512, 512)):
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.input_size = crop_size

        # Paths
        self.images_dir = os.path.join(root_dir, "images", split)
        self.masks_dir = os.path.join(root_dir, "labels", split)

        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))

        # Default image transform
        if self.image_transform is None:
            self.image_transform = T.Compose([
                T.Resize(self.input_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.0918, 0.0918, 0.0918],
                            std=[0.1903, 0.1903, 0.1903])
                # T.Normalize(mean=[0.485, 0.456, 0.406],
                #             std=[0.229, 0.224, 0.225])

            ])

        # Mask resize only (keep as class IDs)
        self.mask_transform = T.Resize(self.input_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- Image ---
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)

        # --- Mask ---
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L')
        mask = self.mask_transform(mask)  # resize first
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # class IDs only

        return image, mask

# class OCTSegDataset2(Dataset):
#     def __init__(self, root_dir, split="train", input_size=(512, 512)):
#         self.root_dir = root_dir
#         self.split = split
#         self.input_size = input_size

#         self.images_dir = os.path.join(root_dir, "images", split)
#         self.masks_dir = os.path.join(root_dir, "labels", split)

#         # Build pairs by matching prefixes
#         image_files = sorted(os.listdir(self.images_dir))
#         mask_files  = sorted(os.listdir(self.masks_dir))

#         # Dict for fast lookup: {prefix: mask_filename}
#         mask_dict = {m.replace("_label.png", ""): m for m in mask_files}

#         self.pairs = []
#         for img in image_files:
#             prefix = img.replace("_image.png", "")
#             if prefix in mask_dict:
#                 self.pairs.append((img, mask_dict[prefix]))
#             else:
#                 print(f"⚠️ No matching mask found for {img}")

#         if split == "train":
#             self.transform = A.Compose([
#                 A.Resize(*input_size),
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.2),
#                 A.RandomRotate90(p=0.5),
#                 A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
#                 A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
#                 A.GaussianBlur(p=0.2),
#                 A.Normalize(mean=(0.485, 0.456, 0.406),
#                             std=(0.229, 0.224, 0.225)),
#                 ToTensorV2()
#             ])
#         else:  # val/test
#             self.transform = A.Compose([
#                 A.Resize(*input_size),
#                 A.Normalize(mean=(0.485, 0.456, 0.406),
#                             std=(0.229, 0.224, 0.225)),
#                 ToTensorV2()
#             ])

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         img_file, mask_file = self.pairs[idx]
#         img_path  = os.path.join(self.images_dir, img_file)
#         mask_path = os.path.join(self.masks_dir, mask_file)

#         image = np.array(Image.open(img_path).convert("RGB"))
#         mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

#         augmented = self.transform(image=image, mask=mask)
#         return augmented["image"], augmented["mask"]

if __name__ == "__main__":
    root = "/media/be-light/Data/PG_Gdansk/Torun_secondment/Experiments/dataset/cityscapes"

    # Train dataset
    train_dataset = OCTSegDataset(root_dir=root, split="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Validation dataset
    val_dataset = OCTSegDataset(root_dir=root, split="val")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Quick check
    img, label = train_dataset[0]
    print("Train Image:", img.shape, "Train Label:", label.shape)
    print("Unique classes in label:", torch.unique(label))
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    # print("Class colors:", train_dataset.mask_transform)
    print("Dataset statistics:")

    for images, masks in train_loader:
        print("Batch:", images.shape, masks.shape)
        break
