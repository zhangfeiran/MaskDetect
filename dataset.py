# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
import math

import random
from torchvision.transforms import functional as FT
import torchvision


# %%
# %%
from lxml import etree
from PIL import ImageDraw, ImageEnhance, ImageFont

# %%
def preprocess():
    print('preprocessing...')
    n = 7959
    dtype = {
        "filename": str,
        "is_train": bool,
        "is_val": bool,
        "width": int,
        "height": int,
        "labels": object,  # 1 for face, 2 for face_mask
        "xmins": object,
        "ymins": object,
        "xmaxs": object,
        "ymaxs": object,
    }
    columns = list(dtype.keys())
    df = pd.DataFrame(index=range(n), columns=columns)

    np.random.seed(0)
    n_train = 6120
    n_val = 839
    n_test = 1000
    vals = set(np.random.choice(n_val + n_test, n_val, False))

    # %%
    idx = 0
    for base in ['./data/AIZOO/train/', './data/AIZOO/val/']:
        is_train = base.endswith('train/')
        for file in sorted(os.listdir(base)):
            if file.endswith('xml'):
                tree = etree.parse(base + file)
                row = [None] * len(columns)
                row[0] = tree.find('filename').text[:-4]
                row[1] = is_train
                row[2] = not is_train and (idx-n_train) in vals
                row[3], row[4] = Image.open(base + file[:-4] + '.jpg').size
                row[5] = [1 if len(i.text) == 4 else 2 for i in tree.iter('name')]
                for i in range(6, 10):
                    row[i] = [int(i.text) for i in tree.iter(columns[i][:-1])]
                df.iloc[idx] = row
                idx += 1
    # %%
    df = df.astype(dtype)
    # %%
    df = df.drop(index=[1927,3888])
    df.index = list(range(len(df)))

    # %%
    df.to_pickle('./data/AIZOO.pkl')

    # %%



# %%
# from dataset import AIZOODataset
class AIZOODataset(torch.utils.data.Dataset):
    def __init__(self, split, transforms, plus=0):
        assert split in {"train", "val", "test"}
        self.plus = plus
        self.split = split
        self.transforms = transforms
        df = pd.read_pickle("./data/AIZOO.pkl")
        if split == "train":
            df = df[df.is_train]
        elif split == "val":
            df = df[~df.is_train & df.is_val]
        else:  # test
            df = df[~df.is_train & ~df.is_val]
        df.index = list(range(len(df)))
        base = ["./data/AIZOO/val/", "./data/AIZOO/train/"][split == "train"]
        self.files = base + df.filename + ".jpg"
        self.df = df

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        boxes = torch.as_tensor(
            [self.df.xmins[idx], self.df.ymins[idx], self.df.xmaxs[idx], self.df.ymaxs[idx]], dtype=torch.float32
        ).t()
        labels = torch.as_tensor(self.df.labels[idx], dtype=torch.int64) + self.plus  # 1 for face, 2 for face_mask retina need minus 1
        return self.transforms(img, boxes, labels, self.split)

    def __len__(self):
        return len(self.files)




def flip(image, boxes):
    image = FT.hflip(image)
    boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
    return image, boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    new_image = FT.resize(image, dims)
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    new_image = image
    distortions = [FT.adjust_brightness, FT.adjust_contrast, FT.adjust_saturation, FT.adjust_hue]
    random.shuffle(distortions)
    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == "adjust_hue":
                adjust_factor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_image = d(new_image, adjust_factor)

    return new_image


def transform_rcnn(image, boxes, labels, split):
    if split == "train":
        image = photometric_distort(image)
        if random.random() < 0.5:
            image, boxes = flip(image, boxes)
    image = FT.to_tensor(image)
    return image, boxes, labels


def transform_ssd(image, boxes, labels, split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        image = photometric_distort(image)
        if random.random() < 0.5:
            image, boxes = flip(image, boxes)

    image, boxes = resize(image, boxes, dims=(300, 300))
    image = FT.to_tensor(image)
    image = FT.normalize(image, mean=mean, std=std)
    return image, boxes, labels

def transform_ssd_no_aug(image, boxes, labels, split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image, boxes = resize(image, boxes, dims=(300, 300))
    image = FT.to_tensor(image)
    image = FT.normalize(image, mean=mean, std=std)
    return image, boxes, labels




def transform_retina(image, boxes, labels, split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        image = photometric_distort(image)
        if random.random() < 0.5:
            image, boxes = flip(image, boxes)

    image, boxes = resize(image, boxes, dims=(512, 512), return_percent_coords=False)
    image = FT.to_tensor(image)
    image = FT.normalize(image, mean=mean, std=std)
    return image, boxes, labels

if __name__ == "__main__":
    preprocess()