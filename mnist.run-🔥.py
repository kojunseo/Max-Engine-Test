import time
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from max import engine


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = datasets.MNIST(download=True, root='./', train=False, transform=transform)

loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)


t1 = time.time()
session = engine.InferenceSession()
input_spec_list = [
    engine.TorchInputSpec(shape=(4, 1, 28, 28), dtype=engine.DType.float32)
]
options = engine.TorchLoadOptions(input_spec_list)
model = session.load("checkpoints/model_script.pt", options)
print("Time to first load session: ", time.time()-t1)


output_target = []
output_trues = []
t2 = time.time()    
for batch in loader:
    outputs = model.execute(x = batch[0])

print("Time to execute model: ", time.time()-t2)
