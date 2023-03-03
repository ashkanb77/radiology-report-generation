from torch.utils.data import Dataset
import os
from PIL import Image


class ImageCaptioningDataset(Dataset):
    def __init__(self, images, reports):
        self.images = images
        self.reports = reports

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join('images', self.images[idx] + '.png')
        image = Image.open(image_path)
        report = self.reports[idx]
        return image, report
