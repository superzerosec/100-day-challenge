# %% [markdown]
# # CNN Image Classification
# 

# %%
# !uv add torchvision


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean but got `align=0`.*")


# %%
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %%
# Hyper-parameters 
num_epochs = 1000
batch_size = 4
learning_rate = 0.001


# %%
# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# %%
# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
X_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

X_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

X_train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,
                                          shuffle=True)

X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %%
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(X_train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))


# %%
conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
fc1 = nn.Linear(16 * 5 * 5, 120)
fc2 = nn.Linear(120, 84)
fc3 = nn.Linear(84, 10)

print(f"CNN input shape: {tuple(images.shape)}")
x = conv1(images)
print(f"Conv1 output shape: {tuple(x.shape)}")
x = pool(x)
print(f"Pool output shape: {tuple(x.shape)}")
x = conv2(x)
print(f"Conv2 output shape: {tuple(x.shape)}")
x = pool(x)
print(f"Pool output shape: {tuple(x.shape)}")


# %%
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


# %%
model = ConvNet().to(device)


# %%
# criterion = nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# %%
n_total_steps = len(X_train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(X_train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


# %%
from pathlib import Path

print('Finished Training')

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "cnn_image_classification.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


# %%
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in X_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')



