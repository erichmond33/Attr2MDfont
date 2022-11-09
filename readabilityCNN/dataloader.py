import os
import random
import pandas as pd

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


class ImageAttr(data.Dataset):
    """Dataset class for the ImageAttr dataset."""
    def __init__(self, image_dir, attr_path, font_readability_path, transform, mode,
                 binary=False, n_style=4,
                 char_num=52, unsuper_num=968, train_num=120, val_num=28):
        """Initialize and preprocess the ImageAttr dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.font_readability_path = font_readability_path
        self.n_style = n_style

        self.transform = transform
        self.mode = mode
        self.binary = binary

        self.super_train_dataset = []
        self.super_test_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.char_num = char_num
        self.unsupervised_font_num = unsuper_num
        self.train_font_num = train_num
        self.val_font_num = val_num

        self.test_super_unsuper = {}
        for super_font in range(self.train_font_num+self.val_font_num):
            self.test_super_unsuper[super_font] = random.randint(0, self.unsupervised_font_num - 1)

        self.char_idx_offset = 10

        self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]

        self.font2readability = {}

        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.super_train_dataset)
        else:
            self.num_images = len(self.super_test_dataset)

    def preprocess(self):
        """Preprocess the font readability csv data."""
        fontReadabilityData = pd.read_csv(self.font_readability_path)

        for i, fontName in enumerate(fontReadabilityData["fontName"]):
            self.font2readability[fontName] = fontReadabilityData["readabilityScore"][i]

        """Preprocess the font attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[1:]

        train_size = self.char_num * self.train_font_num
        val_size = self.char_num * self.val_font_num

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            target_font = filename.split('/')[0]
            target_char = filename.split('/')[1].split('.')[0]
            char_class = int(target_char) - self.char_idx_offset
            font_class = int(i / self.char_num)

            attr_value = []
            for val in values:
                if self.binary:
                    attr_value.append(val == '1')
                else:
                    attr_value.append(eval(val) / 100.0)

            filenameReadabilityScore = self.font2readability[target_font]

            if i < train_size:
                self.super_train_dataset.append([filename, char_class, font_class, attr_value, filenameReadabilityScore])
            elif i < train_size + val_size:
                self.super_test_dataset.append([filename, char_class, font_class, attr_value, filenameReadabilityScore])

        print('Finished preprocessing the Image Attribute (Explo) dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding readability score."""

        if self.mode == 'train':
            if index < len(self.super_train_dataset):
                filename_A, charclass_A, fontclass_A, attr_A, readabilityScore = self.super_train_dataset[index]
        else:
            filename_A, charclass_A, fontclass_A, attr_A, readabilityScore = self.super_test_dataset[index]

        image_A = Image.open(os.path.join(self.image_dir, filename_A)).convert('RGB')

        return {"img_A": self.transform(image_A), "charclass_A": torch.LongTensor([charclass_A]),
                "fontclass_A": torch.LongTensor([fontclass_A]), "attr_A": torch.FloatTensor(attr_A),
                "filename_A" : filename_A, "readabilityScore" : readabilityScore}

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, font_readability_path, image_size=256,
               batch_size=16, dataset_name='explor_all', mode='train', num_workers=0,
               binary=False, n_style=4,
               char_num=52, unsuper_num=968, train_num=120, val_num=28):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset_name == 'explor_all':
        dataset = ImageAttr(image_dir, attr_path, font_readability_path, transform,
                            mode, binary, n_style,
                            char_num=char_num, unsuper_num=unsuper_num,
                            train_num=train_num, val_num=val_num)
    data_loader = data.DataLoader(dataset=dataset,
                                  drop_last=True,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)

    return data_loader
