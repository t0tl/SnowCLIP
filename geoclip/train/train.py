import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader


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

        # Append Queue
        gps_features_q = model.append_gps_queue_features(gps_features, gps)

        img_features_view_1 = torch.unsqueeze(img_features_view_1, 0)
        img_features_view_2 = torch.unsqueeze(img_features_view_2, 0)
        loss = criterion(
            torch.cat((img_features_view_1, img_features_view_2), dim=0),
            gps_features,
            gps_features_q)

        # Backpropagate
        loss.backward()
        optimizer.step()
        batch_loss = loss.item() / batch_size
        epoch_loss += batch_loss
        wandb.log({"train_batch_loss": batch_loss})
        # Update the progress bar
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(train_dataloader)})
    # Update the scheduler
    scheduler.step()
