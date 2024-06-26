from torch import nn
import torch
import wandb
from geopy import distance
import torch.nn.functional as F
import numpy as np
torch.autograd.set_detect_anomaly(True)


class ContrastiveQueueLoss(nn.Module):
    def __init__(self, batch_size, temperature: float = 0.1):
        super(ContrastiveQueueLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    @torch.compile
    def forward(self, V: torch.Tensor, L: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        '''
        V: torch.Tensor
            size (n_aug, batch_size, feature_size)
        L: torch.Tensor
            size (batch_size, feature_size)
        queue: torch.Tensor
            size (queue_size, feature_size)
        '''
        loss = 0
        k = 0
        for uber_i in range(V.shape[1]):
            loss_uber_i = 0
            denominator = 0
            for j in range(V.shape[0]):
                numerator = (V[j, uber_i] @ L[uber_i].T) / self.temperature

                bat_shit_denominator = 0
                for i in range(V.shape[1]):
                    image_vec_list = V[j,i]
                    bat_shit_denominator += torch.exp((image_vec_list @ L[i].T) / self.temperature)

                q_shit_denominator = 0
                for i in range(queue.shape[0]):
                    image_vec_list = V[j, uber_i]
                    q_shit_denominator += torch.exp((image_vec_list @ queue[i].T) / self.temperature)

                denominator = torch.log(bat_shit_denominator + q_shit_denominator)
                loss_uber_i += numerator - denominator
                wandb.log({"numerator": numerator, "denominator": denominator, "loss_uber_i": loss_uber_i})
                k += 1
            loss -= loss_uber_i
        
        loss /= V.shape[1]
        return loss
    
class SnowCLIPLoss(nn.Module):
    def __init__(self, batch_size, queue_size: int, temperature: float = 0.1):
        super(SnowCLIPLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.queue_size = queue_size

    @torch.compile   
    def forward(self, V: torch.Tensor, L: torch.Tensor, gps: torch.Tensor, support_set):#: SupportSetDataStructure) -> torch.Tensor:
        '''
        V: torch.Tensor
            size (n_aug, batch_size, feature_size)
        L: torch.Tensor
            size (batch_size, feature_size)
        support_set: torch.Tensor
            size (support_set_size, feature_size)
        '''
        loss = 0

        for uber_i in range(V.shape[1]):
            loss_uber_i = 0
            denominator = 0
            for j in range(1, V.shape[0]):
                aug_joint = (V[1, uber_i] * L[uber_i].T) 
                aug_joint = F.normalize(aug_joint, dim=0)

                # Get the nearest neighbour for the image in the support set
                org_emb_joint = V[0, uber_i] * L[uber_i]
                org_emb_joint = F.normalize(org_emb_joint, dim=0)
                nn_joint = self.nearest_neighbour(org_emb_joint, support_set)
                nn_joint = F.normalize(nn_joint, dim=0)
                numerator = (nn_joint @ aug_joint) / self.temperature
                batch_denominator = 0
                for i in range(V.shape[1]):
                    aug_joint = (V[j,i] * L[i])
                    aug_joint = F.normalize(aug_joint, dim=0)
                    batch_denominator += torch.exp((nn_joint @ aug_joint) / self.temperature)
                queue_denominator = 0
                negative_features = self.nn_negative_queue(V[0, uber_i], gps[uber_i, :], support_set)
                # TODO: Replace the for loop with a matrix multiplication
                for i in range(negative_features.shape[0]):
                    queue_denominator += torch.exp((nn_joint @ negative_features[i]) / self.temperature)

                # Construct the negative example queue
                denominator = torch.log(batch_denominator + queue_denominator)
                loss_uber_i += numerator - denominator

            loss -= loss_uber_i
        
        loss /= V.shape[1]
        return loss
    
    def nearest_neighbour(self, emb: torch.Tensor, emb_list):#: SupportSetDataStructure) -> torch.Tensor:
        '''
        Find the nearest neighbour in the embedding list
        emb: torch.Tensor (embedding_size)
        emb_list: SupportSetDataStructure
        '''
        # emb_list.support_set["joint_features"] (support_set_size, embedding_size)
        emb = torch.view_copy(emb, (1, emb.shape[0]))

        diagnoal_sim = emb_list.support_set["joint_features"] @ emb.T
        index = torch.argmax(diagnoal_sim)
        return torch.clone(emb_list.support_set["joint_features"][index])

    def nn_negative_queue(self, emb: torch.Tensor, gps: torch.Tensor, emb_list):#: SupportSetDataStructure) -> torch.Tensor:
        '''
        Construct the negative example queue by looking at the nearest neighbour of the image in the support set
        Cutoff 25km.

        Ideas:
        a*distance + b*joint embedding + c*image emb = fp/fn

        Distance > 25 km och cosine sim. joint embedding > 0.5 -> false positive

        Pictures might look very different, leading to less similarity in the embedding space.
        Distance > 25 km och cosine sim. joint embedding > 0.5*(1/cosine sim. img) -> false positive

        Distance > 25 km och cosine sim. joint embedding > 0.5*(a/cosine sim. img) -> false positive, a is a hyperparameter


        QUESTION: WHAT HAPPENS IF WE CHOOSE TOO LARGE VALUE FOR SUPPRORT SET AND WE CAN'T FILL IT? That will show up as a larger loss?
        '''
        false_preds = torch.zeros(self.queue_size, emb_list.support_set["joint_features"].shape[1]).to(emb_list.support_set["joint_features"].device)
        j = 0
        for i in range(emb_list.support_set["gps"].shape[0]):
            lat, lon = emb_list.support_set["gps"][i]
            # NOTE: WIP
            print(gps, (lat, lon))
            if distance.distance(gps, [float(lat), float(lon)]).km > 25:
                false_preds[j] = emb_list.support_set["joint_features"][i]
                j += 1
                if j >= self.queue_size:
                    break
        return false_preds


class SnowCLRLoss(nn.Module):
    def __init__(self, batch_size, queue_size: int, temperature: float = 0.1):
        super(SnowCLRLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.queue_size = queue_size

    def forward(self, V: torch.Tensor, L: torch.Tensor, gps: torch.Tensor, support_set):
        '''
        V: torch.Tensor
            size (n_aug, batch_size, feature_size)
        L: torch.Tensor
            size (batch_size, feature_size)
        support_set: torch.Tensor
            size (support_set_size, feature_size)
        '''
        L_norm = F.normalize(L, dim=1)

        org_emb = V[0] * L_norm  # (batch_size, feature_size) @ (batch_size, feature_size) -> (batch_size, feature_size)
        org_emb_norm = F.normalize(org_emb, dim=1)

        aug_embs = V[1:] * L_norm  # (n_aug-1, batch_size, feature_size) @ (batch_size, feature_size) -> (n_aug-1, batch_size, feature_size)
        aug_embs_norm = F.normalize(aug_embs, dim=2)

        nn_embs = self.nearest_neighbours(org_emb_norm, support_set)
        nn_embs_norm = F.normalize(nn_embs, dim=1)

        # i = n_aug-1, j = batch_size, k = embedding_size 
        similarities = torch.exp(torch.einsum('ijk,ik->ij', aug_embs_norm, nn_embs_norm) / self.temperature)

        batch_denominators = torch.sum(similarities, dim=0)
        
        negative_features = self.nn_negative_queue(org_emb_norm, gps, support_set)  # (batch_size, queue_size, feature_size)
        negative_features_norm = F.normalize(negative_features, dim=2)
        queue_denominators = torch.sum(torch.exp(torch.einsum('ijk,ik->ij', negative_features_norm, nn_embs_norm) / self.temperature), dim=1)
        
        loss = torch.mean(-torch.sum(similarities, dim=0) / (batch_denominators + queue_denominators))

        return loss

    @torch.no_grad
    def nearest_neighbours(self, emb: torch.Tensor, emb_list):#: SupportSetDataStructure) -> torch.Tensor:
        '''
        Find the nearest neighbour in the embedding list
        emb: torch.Tensor (embedding_size)
        emb_list: SupportSetDataStructure
        '''
        # for i in range(emb.shape[0]):
        #     emb_i = torch.view_copy(emb[i], (1, emb[i].shape[0]))
        #     diagnoal_sim = emb_list.support_set["joint_features"] @ emb_i.T
        #     index = torch.argmax(diagnoal_sim)
        #     if i == 0:
        #         nn_embs = torch.clone(emb_list.support_set["joint_features"][index])
        #     else:
        #         nn_embs = torch.cat((nn_embs, emb_list.support_set["joint_features"][index]), dim=0)
        diagnoal_sim = emb_list.support_set["joint_features"] @ emb.T
        diagnoal_sim_dims = diagnoal_sim.shape
        # Diagnoal sim is (support_set_size, batch_size)
        # Take out queue_size nearest neighbours from diagnoal_sim in total
        # Find the indices of the queue_size largest values
        diagnoal_sim = diagnoal_sim.flatten()
        _, flat_indices = torch.topk(diagnoal_sim, self.queue_size)
        row_cols = np.unravel_index(flat_indices.cpu(), diagnoal_sim_dims)
        row = row_cols[0]
        col = row_cols[1]
        # TODO: WIP
        indices = torch.Tensor([(ro, c) for ro, c in zip(row, col)]) #torch.stack((torch.from_numpy(row), torch.from_numpy(col)), dim=0).to(emb_list.support_set["joint_features"].device)
        print(indices.shape)
        print(emb_list.support_set["joint_features"][indices].shape)
        return torch.clone(emb_list.support_set["joint_features"][indices])

    @torch.no_grad
    def nn_negative_queue(self, org_emb_norm: torch.Tensor, gps: torch.Tensor, emb_list):#: SupportSetDataStructure) -> torch.Tensor:
        '''
        Construct the negative example queue by looking at the nearest neighbour of the image in the support set
        Cutoff 25km.

        Ideas:
        a*distance + b*joint embedding + c*image emb = fp/fn

        Distance > 25 km och cosine sim. joint embedding > 0.5 -> false positive

        Pictures might look very different, leading to less similarity in the embedding space.
        Distance > 25 km och cosine sim. joint embedding > 0.5*(1/cosine sim. img) -> false positive

        Distance > 25 km och cosine sim. joint embedding > 0.5*(a/cosine sim. img) -> false positive, a is a hyperparameter


        QUESTION: WHAT HAPPENS IF WE CHOOSE TOO LARGE VALUE FOR SUPPRORT SET AND WE CAN'T FILL IT? That will show up as a smaller loss?
        '''
        false_preds = torch.zeros(self.queue_size, emb_list.support_set["joint_features"].shape[1]).to(emb_list.support_set["joint_features"].device)
        j = 0
        for i in range(emb_list.support_set["gps"].shape[0]):
            lat, lon = emb_list.support_set["gps"][i]
            # NOTE: WIP
            if distance.distance(gps, [float(lat), float(lon)]).km > 25:
                false_preds[j] = emb_list.support_set["joint_features"][i]
                j += 1
                if j >= self.queue_size:
                    break
        return false_preds
