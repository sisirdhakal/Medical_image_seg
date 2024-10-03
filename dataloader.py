import torch
import os
import cv2
from PIL import Image
from helpers import one_hot_encode
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, ToTensor


class BuildingsDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.image_paths = [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]
        self.mask_paths = [
            os.path.join(masks_dir, image_id)
            for image_id in sorted(os.listdir(masks_dir))
        ]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        # return length of
        return len(self.image_paths)

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


def get_dataset(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    pin_memory,
    batch_size,
    num_workers,
    img_height,
    img_width,
):

    train_transform = v2.Compose(
        [
            v2.Resize(height=img_height, width=img_width),
            v2.Rotate(limit=35, p=1.0),
            v2.HorizontalFlip(p=0.5),
            v2.VerticalFlip(p=0.1),
            v2.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
            ToTensor(),
        ],
    )
    test_val_transforms = v2.Compose(
        [
            v2.Resize(height=img_height, width=img_width),
            v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensor(),
        ],
    )

    train_dataset = BuildingsDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        augmentation=train_transform,
    )
    val_dataset = BuildingsDataset(
        images_dir=val_img_dir,
        masks_dir=val_mask_dir,
        augmentation=test_val_transforms,
    )

    trian_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return trian_loader, val_loader
