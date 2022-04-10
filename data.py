import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class SlsDataset(Dataset):
    def __init__(self, sls_dir, nosls_dir, random_dir, transform):
        self.sls_images = sls_dir
        self.nosls_images = nosls_dir
        self.random_images = random_dir
        self.transform = transform
        self.one_hot_encode = {"30": [1.0, 0.0, 0.0, 0.0, 0.0],
                               "50": [0.0, 1.0, 0.0, 0.0, 0.0],
                               "70": [0.0, 0.0, 1.0, 0.0, 0.0],
                               "90": [0.0, 0.0, 0.0, 1.0, 0.0],
                               "110": [0.0, 0.0, 0.0, 0.0, 1.0]}

    def __len__(self):
        return len(self.sls_images) + len(self.nosls_images) + len(self.random_images)

    def __getitem__(self, index):
        if index < len(self.sls_images):
            image = cv2.imread(os.path.join(self.sls_images[index]))
            char_label = self.sls_images[index].split("/")[-1].split("_")[0]
            label = torch.tensor(self.one_hot_encode[char_label])
        elif len(self.sls_images) <= index < len(self.sls_images) + len(self.nosls_images):
            image = cv2.imread(os.path.join(
                self.nosls_images[index - len(self.sls_images)]))
            label = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            image = cv2.imread(os.path.join(
                self.random_images[index - len(self.nosls_images) - len(self.sls_images)]))
            label = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        img = self.transform(image)
        return img, label

    def get_random(self):
        index = np.random.randint(
            0, len(self.nosls_images) + len(self.nosls_images))
        return self.__getitem__(index)

    def viz_random(self):
        index = np.random.randint(
            0, len(self.nosls_images) + len(self.nosls_images))
        image_name = "image of class "
        if index < len(self.sls_images):
            image = cv2.imread(os.path.join(self.sls_images[index]))
            image_name += self.sls_images[index].split("_")[0]
        elif len(self.sls_images) <= index < len(self.sls_images) + len(self.nosls_images):
            image = cv2.imread(os.path.join(
                self.nosls_images[index - len(self.sls_images)]))
            image_name += "not SLS"
        else:
            image = cv2.imread(os.path.join(
                self.random_images[index - len(self.sls_images) - len(self.nosls_images)]))
            image_name += "not SLS"
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
