import torch
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


t1 = time.time()
model = torch.jit.load("checkpoints/res25_script.pt")
print("Time to load model: ", time.time()-t1)


t3 = time.time()
model(torch.randn(4, 25, 3, 224, 224))
print("Time to execute model: ", time.time()-t3)
