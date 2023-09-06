import glob
import random
import os
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, batch_size=None):
        self.transform_hv=transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)])
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.batch_size = batch_size


        self.files_A = sorted(glob.glob(os.path.join(root, 'A') + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'B') + '/*.png'))

    def __getitem__(self, index):


        if  torch.rand(1) < 0.5:

            item_A = self.transform_hv(Image.open(self.files_A[index % len(self.files_A)]))
            item_A=self.transform(item_A)
            item_B = self.transform_hv(Image.open(self.files_B[index % len(self.files_B)]))
            item_B = self.transform(item_B)
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        return {'HE': item_A, 'Ki67': item_B}

    def __len__(self):
        return max(len(self.files_A)//self.batch_size * self.batch_size,
                   len(self.files_B)//self.batch_size * self.batch_size)
class ImageDataset_label(Dataset):
    def __init__(self, root, batch_size=None):
        self.transform_hv=transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)])
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_label = transforms.Compose([
            transforms.ToTensor()])

        self.batch_size = batch_size
        self.files_A = sorted(glob.glob(os.path.join(root, 'A') + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'B') + '/*.png'))
        self.files_C = sorted(glob.glob(os.path.join(root, 'label') + '/*.png'))


    def __getitem__(self, index):
        if torch.rand(1) < 0.5:

            item_A = self.transform_hv(Image.open(self.files_A[index % len(self.files_A)]))
            item_A = self.transform(item_A)
            item_B = self.transform_hv(Image.open(self.files_B[index % len(self.files_B)]))
            item_B = self.transform(item_B)
            item_C = self.transform_hv(Image.open(self.files_C[index % len(self.files_C)]))
            item_C = self.transform_label(item_C)
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            item_C = self.transform_label(Image.open(self.files_C[index % len(self.files_C)]))
        return {'HE': item_A, 'Ki67': item_B, 'Label': item_C}

    def __len__(self):
        return max(len(self.files_A) // self.batch_size * self.batch_size,
                   len(self.files_B) // self.batch_size * self.batch_size)
class ImageDataset_test(Dataset):
    def __init__(self, root, batch_size=None):
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.batch_size = batch_size

        self.files_A = sorted(glob.glob(os.path.join(root, 'A') + '/*.png'))


    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        return {'HE': item_A}

    def __len__(self):
        return len(self.files_A)//self.batch_size * self.batch_size
class ImageDataset_mask(Dataset):
    def __init__(self, root, batch_size=None):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()])

        self.batch_size = batch_size
        self.files_A = sorted(glob.glob(os.path.join(root, 'A') + '/*.png'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'B') + '/*.png'))
        self.files_C = sorted(glob.glob(os.path.join(root, 'mask') + '/*.png'))




    def __getitem__(self, index):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        seed = np.random.randint(2147483647)
        random.seed(seed)
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        seed = np.random.randint(2147483647)
        random.seed(seed)
        item_C = self.transform_mask(Image.open(self.files_C[index % len(self.files_C)]))


        return {'HE': item_A, 'Ki67': item_B, 'Mask':item_C}

    def __len__(self):
        return max(len(self.files_A) // self.batch_size * self.batch_size,
                   len(self.files_B) // self.batch_size * self.batch_size)
