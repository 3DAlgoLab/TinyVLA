# https://huggingface.co/docs/transformers/generation_strategies

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
model.generation_config


# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')

#%%
inputs = tok(["An increasing sequence: one, "], return_tensors='pt')
streamer = TextStreamer(tok)

#%%
_ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
# %%
