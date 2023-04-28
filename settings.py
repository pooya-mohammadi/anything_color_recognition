import os
from dataclasses import dataclass

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class Config:
    device = "cuda"
    dataset_dir = "images"
    output_dir = "output"
    file_name = "best"
    validation_size = 0.2
    batch_size = 64
    train_epochs = 25
    n_workers = 8
    train_lr = 1e-3
    lr_reduce_factor = 0.1
    lr_patience = 5
    input_size = 224
    train_transform = A.Compose(
        [A.Resize(height=input_size, width=input_size),
         A.Rotate(limit=20, p=0.2),
         A.HorizontalFlip(p=0.5),
         A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=255.0),
         ToTensorV2()
         ])
    val_transform = A.Compose(
        [A.Resize(input_size, input_size),
         A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=255.0),
         ToTensorV2()
         ])
    n_classes = 9


Config.n_classes = len(
    [d for d in os.listdir(Config.dataset_dir) if os.path.isdir(os.path.join(Config.dataset_dir, d))])
