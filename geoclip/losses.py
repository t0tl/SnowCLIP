from typing import Tuple, Union
from torch import nn
import torch.nn.functional as F
import torch
import wandb


class ContrastiveQueueLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super(ContrastiveQueueLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, V: torch.Tensor, L: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        '''
        V: torch.Tensor
            size (n_aug, batch_size, feature_size)
        L: torch.Tensor
            size (batch_size, feature_size)
        queue: torch.Tensor
            size (queue_size, feature_size)
        '''
        #pdb.set_trace()
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

class MemoryBankModule(nn.Module):
    """
    Implementation from https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
    """
    def __init__(
            self,
            size: int = 4096,
        ):
            super().__init__()
            self.size = size
            self.bank: torch.Tensor
            self.register_buffer(
                "bank",
                tensor=torch.empty(size=size, dtype=torch.float),
                persistent=False,
            )
            self.bank_ptr: torch.Tensor
            self.register_buffer(
                "bank_ptr",
                tensor=torch.empty(1, dtype=torch.long),
                persistent=False,
            )

    @torch.no_grad()
    def _init_memory_bank(self, size: int) -> None:
        """Initialize the memory bank.

        Args:
            size:
                Size of the memory bank as (num_features, dim) tuple.

        """
        self.bank = torch.randn(size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=-1)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor) -> None:
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """

        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size[0]:
            self.bank[ptr:] = batch[: self.size[0] - ptr].detach()
            self.bank_ptr.zero_()
        else:
            self.bank[ptr : ptr + batch_size] = batch.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(
        self,
        output: torch.Tensor,
        update: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            update:
                If True, the memory bank will be updated with the current output.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank. Entries from the memory bank have
            shape (dim, num_features) if feature_dim_first is True and
            (num_features, dim) otherwise.
        """

        # no memory bank, return the output
        if self.size[0] == 0:
            return output, None

        # Initialize the memory bank if it is not already done.
        if self.bank.ndim == 1:
            dim = output.shape[1:]
            self._init_memory_bank(size=(*self.size, *dim))

        # query and update memory bank
        bank = self.bank.clone().detach()
        if self.feature_dim_first:
            # swap bank size and feature dimension for backwards compatibility
            bank = bank.transpose(0, -1)

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank

class NTXentLoss(MemoryBankModule):
    """        
    Implementation from https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
    """

    def __init__(self, temp: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temp

    def forward(self, img_emb: Tuple[torch.Tensor, torch.Tensor], gps_emb: torch.Tensor, gps_emb_q: torch.Tensor):
        # Forward pass for MOCO inspired setup 
        # with a queue for negative samples
        # x0 and x1 are the augmented images
        # gps_feat_q is the queue of negative samples
        x0, x1 = img_emb

        device = x0.device
        # Normalize the features
        x0 = F.normalize(x0, dim=1)
        x1 = F.normalize(x1, dim=1)

        # This gets the negative samples from the queue
        #x1, negatives = super(NTXentLoss, self).forward(x1, update=x0.required_grad)

        c_0 = x0.dot(gps_emb)
        c_1 = x1.dot(gps_emb)
        sim_pos = c_0.dot(c_1)
        sim_neg = c_0.dot(gps_emb_q) # negatives

        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
        # Zeros implies that the first index is the correct class for the image (the positive sample)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
        return loss

class SimCLRLoss(nn.Module):

    def __init__(self, batch_size: int, temperature: int):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()


    def calc_similarity_batch(self, a: torch.Tensor, b: torch.Tensor):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        
    def forward(self, proj_1: torch.Tensor, proj_2: torch.Tensor):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        # Postive similarities are on the off-diagonals
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        # Mask out the main diagonal and calculate the softmax
        denominator = (self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
