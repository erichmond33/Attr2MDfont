import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader
from model import ReadabilityCNN
from options import get_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(opts):

    # Dirs
    log_dir = os.path.join("readabilityCNN", "experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    samples_dir = os.path.join(log_dir, "samples")
    logs_dir = os.path.join(log_dir, "logs")

    # Path to data
    image_dir = os.path.join("./",opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join("./",opts.data_root, opts.dataset_name, "mdAttributes.txt")
    font_readability_path = os.path.join("./",opts.data_root, "readability.csv")

    test_dataloader = get_loader(image_dir, attribute_path, font_readability_path,
                                    dataset_name="explor_all",
                                    image_size=64,
                                    n_style=4, batch_size=8,
                                    mode='test', binary=False,
                                    train_num=110, val_num=24)

    for batch_idx, batch in enumerate(test_dataloader):
            img_A = batch['img_A'].to(device)
            img_A_Data = img_A.data

            img2 = torch.clone(img_A)
            img3 = torch.clone(img_A)

            img_sample = torch.cat((img_A, img2),dim=1)
            save_file = os.path.join(logs_dir, f"hehe_{batch_idx}.png")
            save_image(img_sample, save_file, nrow=8, normalize=True)

parser = get_parser()
opts = parser.parse_args()

train(opts)