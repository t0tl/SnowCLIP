from model.GeoCLIPSupportSet import GeoCLIPSupportSet
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision.transforms import v2
import numpy as np
import wandb
from datasets import GSV10kDataset, GSVCities
from losses import SnowCLIPLoss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
# WANDB_MODE="disabled"
# os.environ['WANDB_MODE'] = 'disabled'

df_gsv10k = pd.read_csv('/workspace/GSW10k/dataset/coords.csv', header=None)
df_gsv10k.columns = ['lat', 'lon']
df_gsv10k

# Load the gsv-cities dataset
df_barcelona = pd.read_csv("/workspace/gsv-cities/Dataframes/Barcelona.csv").sample(2048, random_state=42)
df_lisbon = pd.read_csv("/workspace/gsv-cities/Dataframes/Lisbon.csv").sample(2048, random_state=42)
#df_madrid = pd.read_csv("/workspace/gsv-cities/Dataframes/Madrid.csv")

df_gsv_cities = pd.concat([df_barcelona, df_lisbon])#, df_madrid])


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
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

EPOCHS = 6
BATCH_SIZE = 128
QUEUE_SIZE = 64
LEARNING_RATE = 1e-4
TEMPERATURE = 0.1
GAMMA = 0.1
STEP_SIZE = 10
K_FOLDS = 3
SUPPORT_SIZE = 1024

criterion = SnowCLIPLoss(BATCH_SIZE, QUEUE_SIZE, temperature=TEMPERATURE)
# dataset = AmsterdamData(root="/workspace/mappilary_street_level/train_val/",
#                         prefix="query/images",
#                         data_df=data_df,
#                         transform=img_transform(),
#                         transform_aug=img_augment_transform())
# dataset = GSV10kDataset(root="/workspace/GSW10k/dataset/",
#                         df=df_gsv10k,
#                         transform=img_transform(),
#                         transform_aug=img_augment_transform())
dataset = GSVCities(root="/workspace/gsv-cities/Images/",
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
def train(train_dataloader: DataLoader,
        model,
        train_iter: int,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        batch_size: int,
        device: str = "cuda:0",
        n_aug: int = 2):
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    epoch_loss = 0
    for i, (imgs, gps) in bar:
        org_imgs = train_transform(imgs)
        views = torch.empty((n_aug,) + org_imgs.shape, dtype=torch.float32)
        views[0] = org_imgs
        for j in range(1, n_aug):
            aug_imgs = aug_train_transform(imgs)
            views[j] = aug_imgs

        imgs = views.to(device)
        gps = gps.to(device)

        optimizer.zero_grad()

        # Forward pass
        img_features_view_1 = model.image_encoder(imgs[0])
        img_features_view_2 = model.image_encoder(imgs[1])
        gps_features = model.location_encoder(gps)

        gps_features = F.normalize(gps_features, dim=1)

        img_features_view_1 = torch.unsqueeze(img_features_view_1, 0)
        img_features_view_2 = torch.unsqueeze(img_features_view_2, 0)
        loss = criterion(
            torch.cat((img_features_view_1, img_features_view_2), dim=0),
            gps_features,
            gps,
            model.support_set)
        # TODO: Keep the support set refreshed, maybe every minibatch
        # or every 10th minibatch'
        # or some percentage of the epoch
        # or after some encoder weight deviation?
        # This will be expensive and probably means we need to keep the support_set size small
        model.update_support_set_features(img_features_view_1.squeeze(dim=0), gps_features, gps)
        # Backpropagate
        loss.backward()
        optimizer.step()
        
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
def test(loader, model, test_iter: int,  criterion, optim, epoch, batch_size, device="cuda:0", test_val="test", n_aug=2):
    epoch_loss = 0
    for i, (imgs, gps) in enumerate(loader):
        optim.zero_grad()

        
        org_imgs = train_transform(imgs)
        views = torch.empty((n_aug,) + org_imgs.shape, dtype=torch.float32)
        views[0] = org_imgs
        for j in range(1, n_aug):
            aug_imgs = aug_train_transform(imgs)
            views[j] = aug_imgs

    
        imgs = views.to(device)
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
        wandb.log({"test_batch_loss": batch_loss, f"{test_val}_iter": test_iter})
        test_iter += 1
    wandb.log({"epoch": epoch, f"{test_val}_loss": epoch_loss / len(loader)})
    return test_iter

@torch.no_grad()
def test_preds(
    loader, model, optim, device="cuda:0"
):
    model.populate_gallery()
    for i, (imgs, gps) in enumerate(loader):
        optim.zero_grad()
        # Make a tensor of 10 views
        views = torch.zeros((10, imgs.shape[0], 3, 224, 224), dtype=torch.float32)
        for j in range(5):
            views[j] = pred_transform(imgs)

        for j in range(5, 10):
            views[j] = pred_transform_flip(imgs)
    
        imgs = views.to(device)
        gps = gps.to(device)

        model.eval_predict(imgs, gps)
    model.delete_gallery()



kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)
for fold, (train_index, test_index) in enumerate(kf.split(train_dataset)):
    train_subset = Subset(dataset, train_index)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    support_set_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    train_iter = 0
    test_iter = 0
    val_iter = 0
    # run.watch(models=snowCLIP, log="all")
    for epoch in range(1, EPOCHS+1):
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="snowclip",
            name=f"eval_SnowCLIP_epoch_{epoch}_fold_{fold}_BATCH_SIZE_{BATCH_SIZE}_QUEUE_SIZE_{QUEUE_SIZE}_SUPPORT_SIZE_{SUPPORT_SIZE}",

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
        snowCLIP = GeoCLIPSupportSet(support_set_loader, batch_size=BATCH_SIZE, device="cuda:0", queue_size=QUEUE_SIZE, support_size=SUPPORT_SIZE, train_transform=train_transform)
        snowCLIP.to("cuda:0")
        os.listdir("finetuned")
        weights = torch.load(f"finetuned/SnowCLIP_{fold}_epoch_{epoch}_BATCH_SIZE_{BATCH_SIZE}_QUEUE_SIZE_{QUEUE_SIZE}_SUPPORT_SIZE_{SUPPORT_SIZE}.pth", map_location="cuda:0")

        # remove "_orig_mod.logit_scale" from the keys
        weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
        snowCLIP.load_state_dict(weights)
        snowCLIP = torch.compile(snowCLIP, mode="reduce-overhead")

        optim = torch.optim.SGD(snowCLIP.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
        train_iter = train(train_loader, snowCLIP, train_iter, criterion, optim, scheduler, epoch=epoch, batch_size=BATCH_SIZE, device="cuda:0")
        #print("Saving model")
        #torch.save(snowCLIP.state_dict(), f"finetuned/SnowCLIP_{fold}_epoch_{epoch}_BATCH_SIZE_{BATCH_SIZE}_QUEUE_SIZE_{QUEUE_SIZE}_SUPPORT_SIZE_{SUPPORT_SIZE}.pth")
        #print("Starting test, epoch:", epoch)
        # Get the test loss for the fold
        test_iter = test(test_loader, snowCLIP, test_iter, criterion, optim, epoch=epoch, batch_size=BATCH_SIZE, device="cuda:0", test_val="test")
        #print("Getting prediction metrics on test set")
        test_preds(test_loader, snowCLIP, optim, BATCH_SIZE, device="cuda:0")
        # Get validation loss
        #val_iter = test(validation_loader, snowCLIP, val_iter, criterion, optim, epoch=epoch, batch_size=BATCH_SIZE, device="cuda:0", test_val="val")
        print("Getting prediction metrics on validation set")
        test_preds(validation_loader, snowCLIP, optim, BATCH_SIZE, device="cuda:0")
    
wandb.finish()