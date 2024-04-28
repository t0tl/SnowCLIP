from model import GeoCLIP
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
import numpy as np
import wandb
from losses import ContrastiveQueueLoss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from datasets import GSVCities
from torchvision import transforms as v2
import os
from tqdm import tqdm
from beartype import beartype

# WANDB_MODE="disabled"
# os.environ['WANDB_MODE'] = 'disabled'

aug_train_transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    v2.RandomGrayscale(p=0.2),
    #v2.PILToTensor(),
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_transform = v2.Compose([
    v2.Resize((224, 224)),
    #v2.PILToTensor(),
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

pred_transform = v2.Compose([
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

pred_transform_flip = v2.Compose([
    v2.RandomHorizontalFlip(p=1),
    v2.ConvertImageDtype(torch.float32),
])

EPOCHS = 6
BATCH_SIZE = 16
QUEUE_SIZE = 128
LEARNING_RATE = 1e-4
TEMPERATURE = 0.1
GAMMA = 0.1
STEP_SIZE = 10
K_FOLDS = 3

criterion = torch.nn.CrossEntropyLoss()

# Load the gsv-cities dataset
df_barcelona = pd.read_csv("/workspaces/SnowCLIP/gsv-cities/Dataframes/Barcelona.csv").sample(2048, random_state=42) #ändra directory
df_lisbon = pd.read_csv("/workspaces/SnowCLIP/gsv-cities/Dataframes/Lisbon.csv").sample(2048, random_state=42)
#df_madrid = pd.read_csv("/workspaces/SnowCLIP/gsv-cities/Dataframes/Madrid.csv")

df_gsv_cities = pd.concat([df_barcelona, df_lisbon])#, df_madrid])

# Load the dataset
dataset = GSVCities(root="/workspaces/SnowCLIP/gsv-cities/Images/", #ändra directory
    df=df_gsv_cities,
    # transform=img_transform(),
    # transform_aug=img_augment_transform()
)


# split dataset into train test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

@torch.enable_grad
@beartype
def train(train_dataloader: DataLoader,
        model,
        train_iter: int,
        criterion: torch.nn.modules.loss._Loss,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        epoch: int,
        batch_size: int,
        device: str = "cuda:0",
        n_aug: int = 2):

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    epoch_loss = 0
    for i, (imgs, gps) in bar:
        targets_img_gps = torch.Tensor([i for i in range(imgs.shape[0])]).long().to(device)
        imgs = aug_train_transform(imgs)
        gps = gps.to(device)
        imgs = imgs.to(device) 
        
        # Forward pass
        optim.zero_grad()

        # Append Queue
        #print("gps shape ", gps.shape)
        gps_q = model.append_gps_queue_features( gps)
        
        logits = model(imgs, gps_q)
        loss = criterion(
            logits,
            targets_img_gps
            )
        # Backpropagate
        loss.backward()
        optim.step()
        batch_loss = loss.item() / batch_size
        epoch_loss += batch_loss
        wandb.log({"train_batch_loss": batch_loss, "train_iter": train_iter})
        # Update the progress bar
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        train_iter += 1

    wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(train_dataloader)})
    # Update the scheduler
    scheduler.step()
    return train_iter


@torch.no_grad()
@beartype
def test(loader, model, test_iter: int, criterion: torch.nn.CrossEntropyLoss, optim, epoch, batch_size, device="cuda:0", test_val="test", n_aug=2):
    epoch_loss = 0

    bar = tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, gps) in bar:
        targets_img_gps = torch.Tensor([i for i in range(imgs.shape[0])]).long().to(device)
        imgs = aug_train_transform(imgs)
        gps = gps.to(device)
        imgs = imgs.to(device) 
        
        # Forward pass
        optim.zero_grad()

        # Append Queue
        #print("gps shape ", gps.shape)
        gps_q = model.append_gps_queue_features( gps)
        
        logits = model(imgs, gps_q)
        loss = criterion(
            logits,
            targets_img_gps
            )
        batch_loss = (loss.item() / batch_size)
        epoch_loss += batch_loss
        wandb.log({"test_batch_loss": batch_loss, "test_iter": test_iter})
        test_iter += 1
    wandb.log({"epoch": epoch, f"{test_val}_loss": epoch_loss / len(loader)})
    return test_iter

@torch.no_grad()
@beartype
def test_preds(
    loader, model, optim, eval_phase: str, device="cuda:0"
):
    model.populate_gallery()
    for batch_number, (imgs, gps) in enumerate(loader):
        optim.zero_grad()
        # Make a tensor of 10 views
        views = torch.zeros((10, imgs.shape[0], 3, 224, 224), dtype=torch.float32)
        for j in range(5):
            views[j] = pred_transform(imgs)

        for j in range(5, 10):
            views[j] = pred_transform_flip(views[j-5])
    
        imgs = views.to(device)
        gps = gps.to(device)

        model.eval_predict(imgs, gps, batch_number, eval_phase)
    model.delete_gallery()


kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)
for j in range(6):
    BATCH_SIZE = 16
    BATCH_SIZE = BATCH_SIZE * (2**(j))


    
    for i in range(6):
        QUEUE_SIZE = 128
        QUEUE_SIZE = QUEUE_SIZE * (2 **(i))
        for fold, (train_index, test_index) in enumerate(kf.split(train_dataset)):
            train_loader = DataLoader(Subset(dataset, train_index), batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(Subset(dataset, test_index), batch_size=BATCH_SIZE, shuffle=True)
            train_iter = 0
            test_iter = 0

            # Load the model
            # weights = torch.load(f"finetuned/geoclip_fold_{fold}_epoch_{epoch}.pth", map_location="cuda:0")

            # remove "_orig_mod.logit_scale" from the keys
            # weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
            geo_clip = GeoCLIP(batch_size=BATCH_SIZE, queue_size=QUEUE_SIZE, device="cuda:0")
            geo_clip.to("cuda:0")
            #geo_clip.load_state_dict(weights)
            #geo_clip = torch.compile(geo_clip, mode='max-autotune', )

            optim = torch.optim.SGD(geo_clip.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
            for epoch in range(1, EPOCHS+1):
                run = wandb.init(
                    # set the wandb project where this run will be logged
                    project="eval_snowclip",
                    name=f"GeoCLIP_epoch_{epoch}_fold_{fold}_batch_{BATCH_SIZE}_queue_{QUEUE_SIZE}",
                    
                    # track hyperparameters and run metadata
                    config={
                    "folds": K_FOLDS,
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "queue_size": QUEUE_SIZE,
                    "dataset": "GSVCities",
                    "architecture": "geoclip",
                    "optimizer": "SGD",
                    "scheduler": {"StepLR": {"step_size": STEP_SIZE, "gamma": GAMMA}},
                    "loss": {"contrastive_queue_loss": {"temperature": TEMPERATURE}},
                    "augmentation": "RandomResizedCrop, RandomHorizontalFlip, RandomApply, RandomGrayscale, ColorJitter",
                    },
                    reinit=True
                )
                run.watch(models=geo_clip, log="all")
                train_iter = train(train_loader, geo_clip, train_iter, criterion, optim, scheduler, epoch=epoch, batch_size=BATCH_SIZE, device="cuda:0")
                torch.save(geo_clip.state_dict(), f"finetuned/geoclip_two_cities_{fold}_epoch_{epoch}_BATCH_SIZE{BATCH_SIZE}_QUEUE_SIZE_{QUEUE_SIZE}.pth")

                print("Starting test, epoch:", epoch)
                test_iter = test(test_loader, geo_clip, train_iter, criterion, optim, epoch=1, batch_size=BATCH_SIZE, device="cuda:0", test_val="test")
                print("Getting test metrics, epoch:", epoch)
                test_preds(test_loader, geo_clip, optim, eval_phase="test")

                print("Getting val metrics, epoch:", epoch)
                test_preds(validation_loader, geo_clip, optim, eval_phase="val")

wandb.finish()