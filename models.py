import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1 ,320)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)

        return x
    

class ResNet25(nn.Module):
    def __init__(self, num_classes=14):
        super(ResNet25, self).__init__()
        self.base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(
            *list(self.base.children())[:-2])
        
        self.dropout_03 = nn.Dropout(0.3)
        self.dropout_05 = nn.Dropout(0.5)

        self.classifier = nn.Linear(2048, num_classes)


    def forward(self, x):
        output = []
        for t in range(x.size(1)):
            output_t = self.feature_extractor(
                x[:, t, :, :, :])  # (b, h, 30, 47)
            output_t = F.avg_pool2d(output_t, (4, 4))
            output_t = output_t.squeeze(3).squeeze(2)  # (b, h)
            output.append(output_t)

        output = torch.stack(output, dim=1)  # (b, n_volumes, h)
        x = output.mean(dim=1)
        x = self.dropout_05(x)
        x_out = self.classifier(x)  # (b, h) -> (b, num_classes)
        return x_out
    