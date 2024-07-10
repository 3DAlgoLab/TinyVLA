# https://huggingface.co/docs/transformers/quicktour
# %%
from transformers import pipeline


#%%

classifier = pipeline("sentiment-analysis", device=0)
classifier("We are very happy to show you the 🤗 Transformers library.")

# %%
classifier = None
# %%
# torch.cuda.empty_cache() # PyTorch thing

transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# %%
