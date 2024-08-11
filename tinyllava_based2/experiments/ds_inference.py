import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
generator = pipeline(
    "text-generation", model="EleutherAI/gpt-neo-2.7B", device=local_rank
)


generator.model = deepspeed.init_inference(
    generator.model,
    tensor_parallel={"tp_size": world_size},
    dtype=torch.float,
    replace_with_kernel_inject=True,
)

string = generator("DeepSpeed is", do_sample=True, min_length=50)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)
