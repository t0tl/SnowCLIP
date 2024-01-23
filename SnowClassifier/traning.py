import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
#from transformers import ViTForImageClassification
#MODEL_CKPT = 'google/vit-base-patch16-224-in21k'

class SnowClassifier(nn.Module):
    def __init__(self):
        super(SnowClassifier, self).__init__()
        # (batch_size, 1, x, y)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # flatten the input
        self.dense = nn.Linear(in_features=28*28*4, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # Input size: (batch_size, 1, x, y)
        x = self.conv1(x) 
        # Output size: (batch_size, 4, x, y)
        x = self.relu(x)
        # Output size: (batch_size, 4, x, y)
        # flatten the input
        x = x.view(x.size(0), -1)
        # Output size: (batch_size, 4 * x * y)
        x = self.dense(x)
        x = self.softmax(x)
        return x
    

m = SnowClassifier()

# pretrained_model = torch.load('model.pth')
# print(pretrained_model.named_parameters())
sgd_optim = SGD(m.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


x_train = torch.randn(100, 1, 128, 128)
y_train = torch.randint(0, 10, (100, 1)).float()
x_test = torch.randn(100, 1, 128, 128)
y_test = torch.randint(0, 10, (100, 1)).float()

# define dataset
# define dataloader
class SnowDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST('./mnist_dataset', download=False, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataset = MNIST('./mnist_dataset', download=False, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

EPOCHS = 10
for ep in range(EPOCHS):
    epoch_loss = 0
    for i, (x, y) in enumerate(train_loader):
        # x = x.to(device)
        # y = y.to(device)
        y_pred = m(x)
        loss = loss_fn(y_pred, y)
        sgd_optim.zero_grad()
        loss.backward()
        sgd_optim.step()
        print(f'Epoch: {ep}, Loss: {loss.item()}')
        epoch_loss += loss.item()

    print(f'Epoch: {ep}, Epoch loss: {epoch_loss/len(train_loader)}')

plt.imshow(x.numpy())
plt.savefig(f'./images/{ep}_y_pred={y_pred}_y={y}.png')
plt.close()
