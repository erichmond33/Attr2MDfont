import datetime
import os
import time

import torch
from torch import nn
# from torchvision.utils import save_image

from readabilityCNN.dataloader import get_loader
from model import CXLoss, DiscriminatorWithClassifier, GeneratorStyle
from options import get_parser
from vgg_cx import VGG19_CX

# Path to data
image_dir = os.path.join("data", "explor_all", "image")
attribute_path = os.path.join("data", "explor_all", "mdAttributes.txt")
readability_path = os.path.join("data", "readability.csv")

# attribute_path = os.path.join("data", "explor_all", "attributes.txt")

# Dataloader
train_dataloader = get_loader(image_dir, attribute_path, readability_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4,
                                batch_size=64, binary=False,
                                train_num=110, val_num=24)
test_dataloader = get_loader(image_dir, attribute_path, readability_path,
                                dataset_name="explor_all",
                                image_size=64,
                                n_style=4, batch_size=8,
                                mode='test', binary=False,
                                train_num=110, val_num=24)
# train_dataloader = get_loader(image_dir, attribute_path,
#                                 dataset_name="explor_all",
#                                 image_size=64,
#                                 n_style=4,
#                                 batch_size=64, binary=False)
# test_dataloader = get_loader(image_dir, attribute_path,
#                                 dataset_name="explor_all",
#                                 image_size=64,
#                                 n_style=4, batch_size=8,
#                                 mode='test', binary=False)

num_epochs = 1
counter = 0
for epoch in range(num_epochs):

    for batch_idx, batch in enumerate(test_dataloader):
        counter += 1
        # if batch_idx >= 3:
        #     break
        # print(" Batch index:", batch_idx)
        # print(" | Batch size:", x.shape[0])
        # print(" | x shape:", x.shape)
        # print(" | y shape:", y.shape)    
        # print(x.keys())
        img_A = batch['img_A']
        print(img_A.shape)      
        print(batch['filename_A'])
        print(batch['charclass_A'])
        print(batch['fontclass_A'])
        print(batch['attr_A'])
        print(batch['readabilityScore'])
        print()
print (counter)
# print("Labels from current batch:", x)