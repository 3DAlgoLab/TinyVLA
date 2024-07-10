#%%
from transformers import pipeline
import torch 

torch.manual_seed(0)
generator = pipeline('text-generation', model='openai-community/gpt2', device_map="auto")

prompt = "hello, i'm a language model"
generator(prompt, max_length=30) 
# %%
text2text_generator = pipeline("text2text-generation", model='google/flan-t5-base')
#%%
prompt = "Translate from English to German: I'm very happy to see you"
text2text_generator(prompt)
# %%
