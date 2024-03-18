import math
import torch    
def forward(self, V: torch.Tensor, L: torch.Tensor, queue: torch.Tensor, support_set: torch.Tensor) -> torch.Tensor:
        '''
        V: torch.Tensor
            size (n_aug, batch_size, feature_size)
        L: torch.Tensor
            size (batch_size, feature_size)
        queue: torch.Tensor
            size (queue_size, feature_size)
        supper_set: torch.Tensor
            size (support_set_size, feature_size)
        '''
        loss = 0

        for uber_i in range(V.shape[1]):
            loss_uber_i = 0
            denominator = 0
            for j in range(1, V.shape[0]):
                origanl_image_location_feature = (V[0, uber_i] *  L[uber_i].T) 
                origanl_image_location_feature = origanl_image_location_feature/origanl_image_location_feature.norm(dim=1)
                nn_vector = NNCLR_list(origanl_image_location_feature, support_set)
                nn_vector = nn_vector / nn_vector.norm(dim=1)
                numerator = torch.exp((V[j, uber_i] @ nn_vector.T) / self.temperature)
                batch_denominator = 0
                for i in range(V.shape[1]):
                    agumented_image_location_feature = (V[j,i] * L[i].T)
                    agumented_image_location_feature = agumented_image_location_feature / agumented_image_location_feature.norm(dim=1)
                    batch_denominator += torch.exp(( nn_vector.T @ agumented_image_location_feature) / self.temperature)

                denominator = torch.log(batch_denominator )
                numerator = torch.log(numerator)
                loss_uber_i += numerator - denominator

            loss -= loss_uber_i
        
        loss /= V.shape[1]
        return loss

def NNCLR_list(origanl_image_location_feature: torch.Tensor, support_set: torch.Tensor) -> torch.Tensor:
    similarity = 0
    nn_vector = origanl_image_location_feature
    for i in range(support_set.shape[0]):
        similarity_value = torch.abs(origanl_image_location_feature @ support_set[i].T)
        if similarity_value > similarity:
            similarity = similarity_value
            nn_vector = support_set[i]

    return nn_vector