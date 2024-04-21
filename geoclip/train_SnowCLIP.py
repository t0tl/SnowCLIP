from model.GeoCLIPSupportSet import GeoCLIPSupportSet
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
import numpy as np
import wandb
import os
from datasets import AmsterdamData, GSV10kDataset
from losses import SnowCLIPLoss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
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

# df_amman = pd.read_csv("/workspace/mappilary_street_level/train_val/amman/query/raw.csv")
# df_amman["city"] = "amman"

# df_saopaolo = pd.read_csv("/workspace/mappilary_street_level/train_val/saopaulo/query/raw.csv")
# df_saopaolo["city"] = "saopaulo"

# df = pd.read_csv("/workspace/mappilary_street_level/train_val/amsterdam/query/raw.csv")
# # Add a column for the city
# df["city"] = "amsterdam"
# # Concatenate df to include all three cities
# df = pd.concat([df_amman, df_saopaolo, df])

# df_no_panorama = df[~df["pano"]]

# df_distance = df_no_panorama[["lat", "lon"]].values
# distances_map = dict()
# min_distance = 100
# start_coords = df_distance[0]
# for i in range(1, len(df_distance)):
#     dist = haversine_distance(start_coords[1], start_coords[0], df_distance[i][1], df_distance[i][0])
#     if dist >= min_distance:
#         start_coords = df_distance[i]
#         distances_map[i] = dist

# selected_images_no_jpg = df_no_panorama.iloc[list(distances_map.keys())].key.values
# selected_images = [f"{img}.jpg" for img in selected_images_no_jpg]
# gps_coords = list(distances_map.keys())

# data_df = df_no_panorama[df_no_panorama["key"].isin(selected_images_no_jpg)]
# data_df["key"] = data_df["key"].apply(lambda x: f"{x}.jpg")
# data_df = data_df.reset_index(drop=True)

# # Filter all rows where key is in selected_images_no_jpg
# df_keys = df_no_panorama[df_no_panorama["key"].isin(selected_images_no_jpg)][["lon", "lat"]]
# np_keys = df_keys.reset_index(drop=True, inplace=False).values

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

EPOCHS = 6
BATCH_SIZE = 128
QUEUE_SIZE = 64
LEARNING_RATE = 1e-4
TEMPERATURE = 0.1
GAMMA = 0.1
STEP_SIZE = 10
K_FOLDS = 3
SUPPORT_SIZE = 256

# Load the model from geoclip_fold_0_epoch_4.pth
# weights = torch.load("finetuned/geoclip_fold_2_epoch_3.pth", map_location="cuda:0")

# remove "_orig_mod.logit_scale" from the keys
# weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}

criterion = SnowCLIPLoss(BATCH_SIZE, QUEUE_SIZE, temperature=TEMPERATURE)
# dataset = AmsterdamData(root="/workspace/mappilary_street_level/train_val/",
#                         prefix="query/images",
#                         data_df=data_df,
#                         transform=img_transform(),
#                         transform_aug=img_augment_transform())
dataset = GSV10kDataset(root="/workspace/GSW10k/dataset/",
                        df=df_gsv10k,
                        transform=img_transform(),
                        transform_aug=img_augment_transform())


# split dataset into train test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

@torch.enable_grad
def train(train_dataloader: DataLoader,
        model,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        batch_size: int,
        device: str = "cuda:0"):
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    epoch_loss = 0
    for i, (imgs, gps) in bar:
        imgs = imgs.to(device)
        gps = gps.to(device)

        optimizer.zero_grad()

        # Forward pass
        img_features_view_1 = model.image_encoder(imgs[:, 0])
        img_features_view_2 = model.image_encoder(imgs[:, 1])
        gps_features = model.location_encoder(gps)

        gps_features = F.normalize(gps_features, dim=1)

        img_features_view_1 = torch.unsqueeze(img_features_view_1, 0)
        img_features_view_2 = torch.unsqueeze(img_features_view_2, 0)
        loss = criterion(
            torch.cat((img_features_view_1, img_features_view_2), dim=0),
            gps_features,
            gps,
            model.support_set)

        # Backpropagate
        loss.backward()
        optimizer.step()
        # TODO: Keep the support set refreshed, maybe every minibatch
        # or every 10th minibatch'
        # or some percentage of the epoch
        # or after some encoder weight deviation?
        # This will be expensive and probably means we need to keep the support_set size small
        # model.update_support_set()
        
        batch_loss = loss.item() / batch_size
        epoch_loss += batch_loss
        wandb.log({"train_batch_loss": batch_loss})
        # Update the progress bar
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(train_dataloader)})
    # Update the scheduler
    scheduler.step()

@torch.no_grad()
def test(loader, model, criterion, optim, epoch, batch_size, device="cuda:0", test_val="test"):
    epoch_loss = 0
    for i, (imgs, gps) in enumerate(loader):
        optim.zero_grad()
        imgs = imgs.to(device)
        gps = gps.to(device)

        # Forward pass
        img_features_view_1 = model.image_encoder(imgs[:, 0])
        img_features_view_2 = model.image_encoder(imgs[:, 1])
        gps_features = model.location_encoder(gps)

        gps_features = F.normalize(gps_features, dim=1)

        img_features_view_1 = torch.unsqueeze(img_features_view_1, 0)
        img_features_view_2 = torch.unsqueeze(img_features_view_2, 0)
        loss = criterion(
            torch.cat((img_features_view_1, img_features_view_2), dim=0),
            gps_features,
            gps,
            model.support_set)
        batch_loss = (loss.item() / batch_size)
        epoch_loss += batch_loss
        wandb.log({"test_batch_loss": batch_loss})
    wandb.log({"epoch": epoch, f"{test_val}_loss": epoch_loss / len(loader)})


kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)
for fold, (train_index, test_index) in enumerate(kf.split(train_dataset)):
    
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="snowclip",

        # track hyperparameters and run metadata
        config={
        "folds": K_FOLDS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "queue_size": QUEUE_SIZE,
        "support_size": SUPPORT_SIZE,
        "dataset": "amsterdam",
        "architecture": "SnowCLIP",
        "optimizer": "SGD",
        "scheduler": {"StepLR": {"step_size": STEP_SIZE, "gamma": GAMMA}},
        "loss": {"contrastive_queue_loss": {"temperature": TEMPERATURE}},
        "augmentations": "RandomResizedCrop, RandomHorizontalFlip, RandomApply, RandomGrayscale, ColorJitter",
        },
        reinit=True
    )
    train_subset = Subset(dataset, train_index)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    support_set_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=BATCH_SIZE, shuffle=False)

    snowCLIP = GeoCLIPSupportSet(support_set_loader, batch_size=BATCH_SIZE, device="cuda:0", queue_size=QUEUE_SIZE, support_size=SUPPORT_SIZE)
    snowCLIP.to("cuda:0")
    #snowCLIP.load_state_dict(weights)
    snowCLIP = torch.compile(snowCLIP, mode='max-autotune')

    optim = torch.optim.SGD(snowCLIP.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
    run.watch(models=snowCLIP, log="all")
    wandb.log({"fold": fold})
    for epoch in range(EPOCHS):
        train(train_loader, snowCLIP, criterion, optim, scheduler, epoch=epoch+1, batch_size=BATCH_SIZE, device="cuda:0")
        print("Starting test, epoch:", epoch+1)
        # Get the test loss for the fold
        test(test_loader, snowCLIP, criterion, optim, epoch=epoch+1, batch_size=BATCH_SIZE, device="cuda:0", test_val="test")
        # Get validation loss
        test(validation_loader, snowCLIP, criterion, optim, epoch=epoch+1, batch_size=BATCH_SIZE, device="cuda:0", test_val="val")
        # Save model
        torch.save(snowCLIP.state_dict(), f"finetuned/SnowCLIP_{fold}_epoch_{epoch+1}.pth")

wandb.finish()