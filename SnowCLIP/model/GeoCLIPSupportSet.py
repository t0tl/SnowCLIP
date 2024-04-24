import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir
import wandb
from geopy import distance

from PIL import Image

class SupportSetDataStructure:
    # Freeze the encoders temporarily
    @torch.no_grad()
    def __init__(self, support_set_loader, image_encoder, location_encoder, batch_size, support_size, train_transform, device="cuda:0"):
        embedding_size = 512
        img_features_ss = torch.zeros(support_size, embedding_size)
        gps_features_ss = torch.zeros(support_size, embedding_size)
        gps_ss = torch.zeros(support_size, 2)
        start_index = 0
        end_index = batch_size
        for i, (imgs, gps) in enumerate(support_set_loader):
            imgs = imgs.to(device)
            gps = gps.to(device)
            # Get the original image, without augmentations
            org_img = train_transform(imgs)
            img_features_view_1 = image_encoder(org_img)
            gps_features = location_encoder(gps)
            gps_features = F.normalize(gps_features, dim=1)
            # If the last batch is smaller than the batch size, repeat the last batch
            # This only happens if we have a support set size that is largert than the number of datapoints in the dataset
            if img_features_view_1.shape[0] != batch_size:
                img_features_view_1 = img_features_view_1.repeat(batch_size, 1)
            if start_index >= support_size:
                break

            img_features_ss[start_index:end_index] = img_features_view_1
            gps_features_ss[start_index:end_index] = gps_features
            gps_ss[start_index:end_index] = gps
            start_index = end_index
            end_index += batch_size

        joint_features = img_features_ss * gps_features_ss
        joint_features = F.normalize(joint_features, dim=1)
        img_features_ss = F.normalize(img_features_ss, dim=1)
        gps_features_ss = F.normalize(gps_features_ss, dim=1)

        self.support_set = dict(img_features=img_features_ss, gps_features=gps_features_ss, gps=gps_ss, joint_features=joint_features)

    def to(self, device):
        """
        Move all tensors to specified device.
        """
        for key, tensor in self.support_set.items():
            self.support_set[key] = tensor.to(device)
        return self

class GeoCLIPSupportSet(nn.Module):
    def __init__(self, support_set_loader, batch_size: int, train_transform, from_pretrained=True, queue_size=128, support_size=2048, device="cuda:0"):
        super().__init__()

        self.batch_size = batch_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        self.support_set_loader = support_set_loader
        self.queue_size = queue_size
        self = self.to(device)

        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "gallery_barcelona_lisbon.csv")).to(device)
        self.support_set = SupportSetDataStructure(support_set_loader, self.image_encoder, self.location_encoder, batch_size, support_size, train_transform)
        self.support_set.to(device)
        self.register_buffer("support_set_ptr", torch.zeros(1, dtype=torch.int32))
        self.support_set_size = support_size

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = device

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))
    
    def update_support_set_features(self, img_emb: torch.Tensor, gps_emb: torch.Tensor, gps_coords: torch.Tensor):
        """
        Replace the oldest support set features with the new ones
        
        Args:
            img_emb (torch.Tensor): Image embeddings
            gps_emb (torch.Tensor): GPS embeddings
            gps_coords (torch.Tensor): GPS coordinates
        """

        gps_batch_size = gps_coords.shape[0]
        ptr = int(self.support_set_ptr)
        new_ptr_position = ptr + gps_batch_size
        assert self.support_set_size % self.batch_size == 0
        # See that the batch size is the same for both image and gps
        assert gps_batch_size == img_emb.shape[0]

        if new_ptr_position > self.support_set_size:
            # Then we will only be able to update part of the support set
            remaining_space_ss = self.support_set_size - ptr
            print(self.support_set.support_set["gps_features"][ptr:].shape, gps_emb.shape, gps_emb[:remaining_space_ss].shape, img_emb[:remaining_space_ss].shape)
            self.support_set.support_set["img_features"][ptr:] = img_emb[:remaining_space_ss]
            self.support_set.support_set["gps_features"][ptr:] = gps_emb[:remaining_space_ss]
            self.support_set.support_set["gps"][ptr:] = gps_coords[:remaining_space_ss]
            ptr = (new_ptr_position) % self.support_set_size
            # Update the start of the array with the remaining elements
            print(self.support_set.support_set["gps_features"][:ptr].shape, img_emb[remaining_space_ss:].shape, gps_emb[remaining_space_ss:].shape)
            self.support_set.support_set["img_features"][:ptr] = img_emb[remaining_space_ss:]
            self.support_set.support_set["gps_features"][:ptr] = gps_emb[remaining_space_ss:]
            self.support_set.support_set["gps"][:ptr] = gps_coords[remaining_space_ss:]
        else:
            self.support_set.support_set["img_features"][ptr:new_ptr_position] = img_emb
            self.support_set.support_set["gps_features"][ptr:new_ptr_position] = gps_emb
            self.support_set.support_set["gps"][ptr:new_ptr_position] = gps_coords
        ptr = (new_ptr_position) % self.support_set_size  # move pointer
        self.support_set_ptr[0] = ptr

                                             
    def forward(self, image, location):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Scalar product (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features)

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return
        """
        image = Image.open(image_path)
        image = ImageEncoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k prediction
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob
    
    @torch.no_grad()
    def populate_gallery(self):
        self.gallery_embs = self.location_encoder(self.gps_gallery)

    @torch.no_grad()
    def delete_gallery(self):
        del self.gallery_embs

    @torch.no_grad()
    def eval_predict(self, images, gps):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return
        """
        images = images.to(self.device)
        logit_scale = self.logit_scale.exp()

        # Store the image embeddings in a tesnor (n_aug, batch_size, 512)
        all_img_embs = torch.zeros(images.shape[0], images.shape[1], 512).to(self.device)
        
        for i in range(images.shape[0]):
            img_embs = self.image_encoder(images[i])
            all_img_embs[i] = img_embs

        logits_per_img = logit_scale * (all_img_embs @ self.gallery_embs.T)

        # (n_aug, batch_size, 100_000)
        probs_per_image = logits_per_img.softmax(dim=0).cpu()

        top_pred = torch.argmax(probs_per_image, dim=2)
        top_pred_gps = self.gps_gallery[top_pred].cpu()
        for i in range(top_pred_gps.shape[1]):
            error = 0
            correct_city = False
            avg_pred = top_pred_gps.mean(dim=0)
            for j in range(top_pred_gps.shape[0]):
                error += distance.distance(top_pred_gps[j, i], gps[i]).km

            if distance.distance(avg_pred[i], gps[i]).km < 25:
                correct_city = True
            error /= top_pred_gps.shape[0]
            wandb.log({"mean_error_distance": error, "gps": gps[i], "avg_pred": avg_pred[i], "correct_city": correct_city})