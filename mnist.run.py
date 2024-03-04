import torch
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = datasets.MNIST(download=True, root='./', train=False, transform=transform)

loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)


t1 = time.time()
model = torch.jit.load("checkpoints/model_script.pt")
print("Time to load model: ", time.time()-t1)



output_target = []
output_trues = []
t3 = time.time()
for batch in loader:
    outputs = model(batch[0])
    output_classes = outputs.argmax(dim=1)
    output_target.append(output_classes)
    output_trues.append(batch[1])
print("Time to execute model: ", time.time()-t3)
print("Accuracy: ", (torch.cat(output_target) == torch.cat(output_trues)).float().mean().item())