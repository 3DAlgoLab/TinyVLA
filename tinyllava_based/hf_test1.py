# %%

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from icecream import ic

hf_path = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
model = AutoModelForCausalLM.from_pretrained(
    hf_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.cuda()
ic(model)

config = model.config
ic(config)

None

# %%
tokenizer = AutoTokenizer.from_pretrained(
    hf_path,
    use_fast=False,
    model_max_length=config.tokenizer_model_max_length,
    padding_side=config.tokenizer_padding_side,
)
prompt = "What are these?"
image_url = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
output_text, generation_time = model.chat(
    prompt=prompt, image=image_url, tokenizer=tokenizer
)

print("model output:", output_text)
print("running time:", generation_time)

# %%
