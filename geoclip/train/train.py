import torch
from tqdm import tqdm
import torch.nn.functional as F

def train(train_dataloader, model, criterion, optimizer, scheduler, epoch, batch_size, device):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).to(device).double()

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

        loss = 0
        img_gps_loss = criterion(
            (img_features_view_1, img_features_view_2), gps_features, gps_features_q)
        loss += img_gps_loss

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Update the progress bar
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    # Update the scheduler
    scheduler.step()
