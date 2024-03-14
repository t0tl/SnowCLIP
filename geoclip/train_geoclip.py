from model import GeoCLIP
from model import ImageEncoder
from model import LocationEncoder
from train import train
import dataloader
import os
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import wandb
import os
from datasets import AmsterdamData, GSV10kDataset
from losses import ContrastiveQueueLoss
import requests
from transformers import AutoProcessor, CLIPModel
# WANDB_MODE="disabled"
# os.environ['WANDB_MODE'] = 'disabled'

def haversine_distance(lon1, lat1, lon2, lat2):
    '''Find distance between locations in meters'''

    R = 6371000 # radius of Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2))**2 + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2))**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df_gsv10k = pd.read_csv('/workspace/GSW10k/dataset/coords.csv', header=None)
df_gsv10k.columns = ['lat', 'lon']
df_gsv10k

df_amman = pd.read_csv("/workspace/mappilary_street_level/train_val/amman/query/raw.csv")
df_amman["city"] = "amman"

df_saopaolo = pd.read_csv("/workspace/mappilary_street_level/train_val/saopaulo/query/raw.csv")
df_saopaolo["city"] = "saopaulo"

df = pd.read_csv("/workspace/mappilary_street_level/train_val/amsterdam/query/raw.csv")
# Add a column for the city
df["city"] = "amsterdam"
# Concatenate df to include all three cities
df = pd.concat([df_amman, df_saopaolo, df])

df_no_panorama = df[~df["pano"]]

df_distance = df_no_panorama[["lat", "lon"]].values
distances_map = dict()
min_distance = 100
start_coords = df_distance[0]
for i in range(1, len(df_distance)):
    dist = haversine_distance(start_coords[1], start_coords[0], df_distance[i][1], df_distance[i][0])
    if dist >= min_distance:
        start_coords = df_distance[i]
        distances_map[i] = dist

selected_images_no_jpg = df_no_panorama.iloc[list(distances_map.keys())].key.values
selected_images = [f"{img}.jpg" for img in selected_images_no_jpg]
gps_coords = list(distances_map.keys())

data_df = df_no_panorama[df_no_panorama["key"].isin(selected_images_no_jpg)]
data_df["key"] = data_df["key"].apply(lambda x: f"{x}.jpg")
data_df = data_df.reset_index(drop=True)

# Filter all rows where key is in selected_images_no_jpg
df_keys = df_no_panorama[df_no_panorama["key"].isin(selected_images_no_jpg)][["lon", "lat"]]
np_keys = df_keys.reset_index(drop=True, inplace=False).values

def img_augment_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_transform():
    train_transform_list = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

EPOCHS = 2
BATCH_SIZE = 256
QUEUE_SIZE = 2048

geo_clip = GeoCLIP(batch_size=BATCH_SIZE, device="cuda:0", queue_size=QUEUE_SIZE)
geo_clip.to("cuda:0")


LEARNING_RATE = 1e-4
TEMPERATURE = 0.1
GAMMA = 0.1
STEP_SIZE = 10
optim = torch.optim.SGD(geo_clip.parameters(), lr=LEARNING_RATE)
criterion = ContrastiveQueueLoss(batch_size=BATCH_SIZE, temperature=TEMPERATURE)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
dataset = AmsterdamData(root="/workspace/mappilary_street_level/train_val/",
                        prefix="query/images",
                        data_df=data_df,
                        transform=img_transform(),
                        transform_aug=img_augment_transform())
dataset = GSV10kDataset(root="/workspace/GSW10k/dataset/",
                        df=df_gsv10k,
                        transform=img_transform(),
                        transform_aug=img_augment_transform())


# split dataset into train test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

run = wandb.init(
    # set the wandb project where this run will be logged
    project="snowclip",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "queue_size": QUEUE_SIZE,
    "dataset": "amsterdam",
    "architecture": "geoclip",
    "optimizer": "SGD",
    "scheduler": {"StepLR": {"step_size": STEP_SIZE, "gamma": GAMMA}},
    "loss": {"contrastive_queue_loss": {"temperature": TEMPERATURE}},
    "augmentation": "RandomResizedCrop, RandomHorizontalFlip, RandomApply, RandomGrayscale",
    }
)


@torch.no_grad()
def test(loader, model, criterion, optim, scheduler, epoch, batch_size, device="cuda:0"):
    epoch_loss = 0
    for i, (imgs, gps) in enumerate(loader):
        optim.zero_grad()
        imgs = imgs.to(device)
        gps = gps.to(device)

        img_features_view_1 = model.image_encoder(imgs[:, 0])
        img_features_view_2 = model.image_encoder(imgs[:, 1])
        gps_features = model.location_encoder(gps)
        gps_features = F.normalize(gps_features, dim=1)

        # Append Queue
        gps_features_q = model.append_gps_queue_features(gps_features, gps)

        img_features_view_1 = torch.unsqueeze(img_features_view_1, 0)
        img_features_view_2 = torch.unsqueeze(img_features_view_2, 0)
        loss = criterion(
            torch.cat((img_features_view_1, img_features_view_2), dim=0),
            gps_features,
            gps_features_q)
        batch_loss = (loss.item() / batch_size)
        epoch_loss += batch_loss
        wandb.log({"test_batch_loss": batch_loss})
    wandb.log({"epoch": epoch, "test_loss": epoch_loss / len(loader)})


run.watch(models=geo_clip, criterion=criterion, log="all", log_freq=1)
for epoch in range(EPOCHS):
    train(train_loader, geo_clip, criterion, optim, scheduler, epoch=epoch+1, batch_size=BATCH_SIZE, device="cuda:0")
    print("Starting test, epoch:", epoch)
    test(test_loader, geo_clip, criterion, optim, scheduler, epoch=epoch+1, batch_size=BATCH_SIZE, device="cuda:0")
wandb.finish()