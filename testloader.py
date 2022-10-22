import datetime
import os
import time

import torch
from torch import nn
# from torchvision.utils import save_image

from dataloader import get_loader
from model import CXLoss, DiscriminatorWithClassifier, GeneratorStyle
from options import get_parser
from vgg_cx import VGG19_CX

# Path to data
image_dir = os.path.join("data", "explor_all", "image")
attribute_path = os.path.join("data", "explor_all", "attributes.txt")

# Dataloader
train_dataloader = get_loader(image_dir, attribute_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4,
                                batch_size=64, binary=False)
test_dataloader = get_loader(image_dir, attribute_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4, batch_size=8,
                                mode='test', binary=False)

num_epochs = 1
for epoch in range(num_epochs):

    for batch_idx, x in enumerate(test_dataloader):
        if batch_idx >= 3:
            break
        # print(" Batch index:", batch_idx)
        # print(" | Batch size:", x.shape[0])
        # print(" | x shape:", x.shape)
        # print(" | y shape:", y.shape)    
        print(x.keys())        

# print("Labels from current batch:", x)