import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 10
}

dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, config["batch_size"], shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(x)

model = MLP()
model.to(config["device"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

losses = []
accuracies = []

for epoch in range(config["epochs"]):
    running_loss = 0.0
    running_accuracies = 0.0
    for imgs, labels in dataloader:
        imgs = imgs.view(imgs.size(0), -1)
        imgs = imgs.to(config["device"])
        labels = labels.to(config["device"])
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        running_accuracies += torch.mean(pred.eq(labels).float())
        loss.backward()
        optimizer.step()
    running_loss /= len(dataloader)
    running_accuracies /= len(dataloader)
    losses.append(running_loss)
    accuracies.append(running_accuracies)
    print("epoch : ", epoch, "loss : ", running_loss, "accuracy : ", running_accuracies)





