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
        # See that the batch size is the same for both image and gps
        assert gps_batch_size == img_emb.shape[0]

        if new_ptr_position > self.support_set_size:
            # Then we will only be able to update part of the support set
            remaining_space_ss = self.support_set_size - ptr
            self.support_set.support_set["img_features"][ptr:] = img_emb[:remaining_space_ss]
            self.support_set.support_set["gps_features"][ptr:] = gps_emb[:remaining_space_ss]
            self.support_set.support_set["gps"][ptr:] = gps_coords[:remaining_space_ss]
            ptr = (new_ptr_position) % self.support_set_size
            # Update the start of the array with the remaining elements
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
    def eval_sum(self, probs_per_image: torch.Tensor, gps: torch.Tensor, batch_number: int, eval_phase: str):
        sum_probs = probs_per_image.sum(dim=0)
        top_pred_sum = torch.argmax(sum_probs, dim=1)
        top_pred_gps_sum = self.gps_gallery[top_pred_sum].cpu()
        batch_error = 0
        batch_city_accuracy = 0
        batch_street_accuracy = 0
        for i in range(top_pred_gps_sum.shape[0]):
            error = distance.distance(top_pred_gps_sum[i], gps[i]).km
            batch_error += error
            correct_city = False
            correct_street = False

            if error < 25:
                correct_city = True
                batch_city_accuracy += 1

            if error < 1:
                correct_street = True
                batch_street_accuracy += 1

            # Log raw data
            wandb.log({
                f"{eval_phase}_sum_error_distance": error,
                f"{eval_phase}_gps": gps[i],
                f"{eval_phase}_sum_avg_pred": top_pred_gps_sum[i],
                f"{eval_phase}_sum_correct_city": correct_city,
                f"{eval_phase}_sum_correct_street": correct_street
            })
        
        batch_error /= top_pred_gps_sum.shape[0]
        batch_city_accuracy /= top_pred_gps_sum.shape[0]
        batch_street_accuracy /= top_pred_gps_sum.shape[0]
        wandb.log({
            f"{eval_phase}_batch_number": batch_number,
            f"{eval_phase}_batch_sum_error_distance": batch_error,
            f"{eval_phase}_batch_sum_correct_city_acc": batch_city_accuracy,
            f"{eval_phase}_batch_sum_correct_street_acc": batch_street_accuracy
        })
        
    @torch.no_grad()
    def eval_mean(self, top_pred: torch.Tensor, gps: torch.Tensor, batch_number: int, eval_phase: str):
        top_pred_gps = self.gps_gallery[top_pred].cpu()
        batch_mean_error_distance = 0
        batch_mean_correct_city_acc = 0
        batch_mean_correct_street_acc = 0
        batch_majority_vote_acc = 0
        for i in range(top_pred_gps.shape[1]):
            error = 0
            correct_city = False
            correct_street = False
            majority_vote = 0

            avg_pred = top_pred_gps.mean(dim=0)
            for j in range(top_pred_gps.shape[0]):
                diff = distance.distance(top_pred_gps[j, i], gps[i]).km
                error += diff
                if diff < 25:
                    majority_vote += 1

            avg_distance_error = distance.distance(avg_pred[i], gps[i]).km
            if avg_distance_error < 25:
                correct_city = True
                batch_mean_correct_city_acc += 1

            if avg_distance_error < 1:
                correct_street = True
                batch_mean_correct_street_acc += 1

            error /= top_pred_gps.shape[0]
            batch_mean_error_distance += error

            majority_vote_city = False
            # If half the predictions are within 25 km then we got the correct city
            if majority_vote >= (top_pred_gps.shape[0] / 2):
                majority_vote_city = True
                batch_majority_vote_acc += 1

            wandb.log({
                f"{eval_phase}_mean_error_distance": error,
                f"{eval_phase}_gps": gps[i],
                f"{eval_phase}_avg_pred": avg_pred[i],
                f"{eval_phase}_correct_city": correct_city,
                f"{eval_phase}_correct_street": correct_street,
                # If the majority of the predictions are within 25 km then we got the correct city
                f"{eval_phase}_majority_vote": majority_vote_city,
                f"{eval_phase}_avg_error_distance": avg_distance_error
            })
        
        batch_mean_error_distance /= top_pred_gps.shape[1]
        batch_mean_correct_city_acc /= top_pred_gps.shape[1]
        batch_mean_correct_street_acc /= top_pred_gps.shape[1]
        batch_majority_vote_acc /= top_pred_gps.shape[1]

        wandb.log({
            f"{eval_phase}_batch_number": batch_number,
            f"{eval_phase}_batch_mean_error_distance": batch_mean_error_distance,
            f"{eval_phase}_batch_mean_correct_city_acc": batch_mean_correct_city_acc,
            f"{eval_phase}_batch_mean_correct_street_acc": batch_mean_correct_street_acc,
            f"{eval_phase}_batch_majority_vote_acc": batch_majority_vote_acc
        })

    @torch.no_grad()
    def eval_emb_mean(self, top_pred: torch.Tensor, gps: torch.Tensor, batch_number: int, eval_phase: str):
        pred_embeddings = self.gallery_embs[top_pred]
        avg_embedding = pred_embeddings.mean(dim=0)
        # Similarity between the average embedding and the embedding gallery
        # Shape (batch_size, emb_size), (gallery_size, emb_size) -> (batch_size, gallery_size)
        logits = avg_embedding @ self.gallery_embs.T
        probs_per_image = logits.softmax(dim=0).cpu()
        top_pred = torch.argmax(probs_per_image, dim=1)
        top_pred_gps = self.gps_gallery[top_pred].cpu()
        batch_emb_mean_error = 0
        batch_emb_mean_city_acc = 0
        batch_emb_mean_street_acc = 0
        for i in range(top_pred_gps.shape[0]):
            error = distance.distance(top_pred_gps[i], gps[i]).km
            batch_emb_mean_error += error
            correct_city = False
            correct_street = False

            if error < 25:
                correct_city = True
                batch_emb_mean_city_acc += 1

            if error < 1:
                correct_street = True
                batch_emb_mean_street_acc += 1

            wandb.log({
                f"{eval_phase}_emb_mean_error_distance": error,
                f"{eval_phase}_gps": gps[i],
                f"{eval_phase}_emb_avg_pred": top_pred_gps[i],
                f"{eval_phase}_emb_correct_city": correct_city,
                f"{eval_phase}_emb_correct_street": correct_street
            })
        batch_emb_mean_error /= top_pred_gps.shape[0]
        batch_emb_mean_city_acc /= top_pred_gps.shape[0]
        batch_emb_mean_street_acc /= top_pred_gps.shape[0]
        wandb.log({
            f"{eval_phase}_batch_number": batch_number,
            f"{eval_phase}_batch_emb_mean_error": batch_emb_mean_error,
            f"{eval_phase}_batch_emb_mean_city_acc": batch_emb_mean_city_acc,
            f"{eval_phase}_batch_emb_mean_street_acc": batch_emb_mean_street_acc
        })

    def eval_logit_mean(self, logits_per_img, gps, batch_number: int, eval_phase: str):
        mean_logits = logits_per_img.mean(dim=0)
        mean_probs = mean_logits.softmax(dim=1).cpu()
        top_pred = torch.argmax(mean_probs, dim=1)
        top_pred_gps = self.gps_gallery[top_pred].cpu()
        batch_error = 0
        batch_city_accuracy = 0
        batch_street_accuracy = 0
        for i in range(top_pred_gps.shape[0]):
            error = distance.distance(top_pred_gps[i], gps[i]).km
            batch_error += error
            correct_city = False
            correct_street = False

            if error < 25:
                correct_city = True
                batch_city_accuracy += 1

            if error < 1:
                correct_street = True
                batch_street_accuracy += 1

            # Log raw data
            wandb.log({
                f"{eval_phase}_logit_mean_error_distance": error,
                f"{eval_phase}_gps": gps[i],
                f"{eval_phase}_logit_mean_avg_pred": top_pred_gps[i],
                f"{eval_phase}_logit_mean_correct_city": correct_city,
                f"{eval_phase}_logit_mean_correct_street": correct_street
            })
        
        batch_error /= top_pred_gps.shape[0]
        batch_city_accuracy /= top_pred_gps.shape[0]
        batch_street_accuracy /= top_pred_gps.shape[0]
        wandb.log({
            f"{eval_phase}_batch_number": batch_number,
            f"{eval_phase}_batch_logit_mean_error_distance": batch_error,
            f"{eval_phase}_batch_logit_mean_correct_city_acc": batch_city_accuracy,
            f"{eval_phase}_batch_logit_mean_correct_street_acc": batch_street_accuracy
        })




    @torch.no_grad()
    def eval_predict(self, images, gps, batch_number: int, eval_phase: str):
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

        # (n_aug, batch_size, gallery_size)
        self.eval_logit_mean(logits_per_img, gps, batch_number, eval_phase)

        probs_per_image = logits_per_img.softmax(dim=0).cpu()
        self.eval_sum(probs_per_image, gps, batch_number, eval_phase)
        # top_pred = torch.argmax(probs_per_image, dim=2)
        # self.eval_mean(top_pred, gps, batch_number, eval_phase)

        # self.eval_emb_mean(top_pred, gps, batch_number, eval_phase)
