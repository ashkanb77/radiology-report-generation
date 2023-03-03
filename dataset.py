from torch.utils.data import Dataset
from PIL import Image


class ImageCaptioningDataset(Dataset):
    def __init__(self, images, reports):
        self.images = images
        self.reports = reports

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        report = self.reports[idx]
        return image, report
