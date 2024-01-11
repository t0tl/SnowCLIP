import torch
from torch import nn
from transformers import ViTForImageClassification


MODEL_CKPT = 'google/vit-base-patch16-224-in21k'

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
    

m = TestModel()
m.load_state_dict(torch.load('pytorch_model.bin')) 
