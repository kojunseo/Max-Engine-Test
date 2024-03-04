import numpy as np
import torch

from models import ConvNet


convNet = ConvNet()

model_path = "checkpoints/model_torch.pth"
dict_model = torch.load(model_path)
convNet.load_state_dict(dict_model['net'])

with torch.no_grad():
        traced_model = torch.jit.trace(
            convNet, strict=False, example_inputs=torch.randn(4, 1, 28, 28)
        )

torch.jit.save(traced_model, "checkpoints/model_script.pt")

