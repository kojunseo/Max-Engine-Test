import numpy as np
import torch

from models import ResNet25


convNet = ResNet25()

model_path = "checkpoints/res25_torch.pth"
convNet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

convNet.eval()

with torch.no_grad():
    traced_model = torch.jit.trace(
        convNet, strict=True, example_inputs=torch.randn(4, 25, 3, 224, 224)
    )

torch.jit.save(traced_model, "checkpoints/res25_script.pt")

