from torch.utils.data import DataLoader
import numpy as np
import torchdata.datapipes as dp
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None):

        df = pd.read_csv(csv_path)
        self.transform = transform

        # based on DataFrame columns
        self.font_names = df["fontId"]
        self.averages = df["average"]
        self.more = df["average"]
        self.mores = df["average"]

    def __getitem__(self, index):

        averages = self.averages[index]
        more = self.more[index]
        mores = self.mores[index]
        return {"a" : averages, "m" : more, "m2" : mores}

    def __len__(self):
        return self.averages.shape[0]

train_dataset = MyDataset(
    csv_path="./data/readability.csv"
    # transform=data_transforms["train"],
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    # transform=data_transforms["train"],
    num_workers=0,
)

num_epochs = 1
for epoch in range(num_epochs):

    for batch_idx, x in enumerate(train_loader):
        # if batch_idx >= 3:
        #     break
        # print(" Batch index:", batch_idx)
        # print(" | Batch size:", x.shape[0])
        # print(" | x shape:", x.shape)
        # print(" | y shape:", x[""])  
        print(x.keys())             

print("Labels from current batch:", x)

# def filter_for_data(filename):
#     return "sample_data" in filename and filename.endswith(".csv")

# def row_processer(row):
#     return {"label": np.array(row[0], np.int32), "data": np.array(row[1:], dtype=np.float64)}

# def build_datapipes(root_dir="./data"):
#     datapipe = dp.iter.FileLister(root_dir)
#     datapipe = datapipe.filter(filter_fn=filter_for_data)
#     datapipe = datapipe.open_files(mode='rt')
#     datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
#     # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
#     datapipe = datapipe.shuffle()
#     datapipe = datapipe.map(row_processer)
#     return datapipe

# if __name__ == '__main__':
#     datapipe = build_datapipes()
#     dl = DataLoader(dataset=datapipe, batch_size=5, num_workers=2)
#     first = next(iter(dl))
#     labels, features = first['label'], first['data']
#     print(f"Labels batch shape: {labels.size()}")
#     print(f"Feature batch shape: {features.size()}")
#     print(f"{labels = }\n{features = }")
#     n_sample = 0
#     for row in iter(dl):
#         n_sample += 1
#     print(f"{n_sample = }")