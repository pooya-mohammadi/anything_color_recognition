import torch
from torch.utils.data import Dataset
import cv2


class VehicleDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None, class_to_id=None):
        self.images = image_list
        self.labels = label_list
        self.transform = transform
        self.class_to_id = class_to_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.images[idx]
        try:
            img = cv2.imread(image_path)[..., ::-1]  # bgr2rgb
        except:
            raise ValueError(image_path)
        if self.transform:
            img = self.transform(image=img)["image"]
        label_name = self.labels[idx]
        label = torch.tensor(self.class_to_id[label_name]).type(torch.long)
        sample = (img, label)
        return sample
