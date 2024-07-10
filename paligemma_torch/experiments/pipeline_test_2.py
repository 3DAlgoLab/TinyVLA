# pip install accelerate
# ref: https://huggingface.co/docs/transformers/pipeline_tutorial#multimodal-pipeline
#%%
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
# %%
print(output)
# %%
pipe=None

# %%
import gc 
gc.collect() 
torch.cuda.empty_cache()
# %%
