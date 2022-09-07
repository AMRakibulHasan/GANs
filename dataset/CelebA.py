from torch.utils.data import Dataset
from dataset.util import Transform
from PIL import Image

"""
CelebA数据集
"""


class CelebAData(Dataset):
    def __init__(self, data_path, img_size):
        self.data_path = data_path
        self.tf = Transform(img_size)
        self.file = ['%s.jpg' % i for i in range(30000)]

    def __getitem__(self, item):
        img = Image.open(self.data_path + self.file[item])
        img = self.tf(img)

        return img

    def __len__(self):
        return len(self.file)
