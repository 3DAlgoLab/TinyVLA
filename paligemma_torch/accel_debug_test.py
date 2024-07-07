import torch
from accelerate import PartialState
from accelerate.utils import broadcast

state = PartialState()
if state.process_index == 0:
    tensor = torch.tensor([[0.0, 1, 2, 3, 4]]).to(state.device)
else:
    tensor = torch.tensor([[[0.0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]).to(state.device)

broadcast_tensor = broadcast(tensor)
print(broadcast_tensor)
