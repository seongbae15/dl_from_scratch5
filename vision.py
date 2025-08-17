import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

for x, label in dataloader:
    print(x.shape)
    print(label.shape)
    break
