import time
import torch


from max import engine


t1 = time.time()
session = engine.InferenceSession()
input_spec_list = [
    engine.TorchInputSpec(shape=(4, 25, 3, 224, 224), dtype=engine.DType.float32)
]
options = engine.TorchLoadOptions(input_spec_list)
model = session.load("checkpoints/res25_script.pt", options)
print("Time to first load session: ", time.time()-t1)

t2 = time.time() 
model.execute(x = torch.randn(4, 25, 3, 224, 224))
print("Time to execute model: ", time.time()-t2)
