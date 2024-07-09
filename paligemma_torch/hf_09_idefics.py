# https://huggingface.co/docs/transformers/tasks/idefics

#%%
import torch 
from transformers import IdeficsForVisionText2Text, AutoProcessor

checkpoint = "HuggingFaceM4/idefics-9b"
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map='auto')


# %%
prompt = [
    "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
]
inputs = processor(prompt, return_tensors="pt").to("cuda")

bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])

#%%
prompt = [
    "https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/e0fbf80d722c31ca1c26696cf1d3b75db6511f79fcd2f3df1ec33be1ed524011?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27idefics-prompted-im-captioning.jpg%3B+filename%3D%22idefics-prompted-im-captioning.jpg%22%3B&response-content-type=image%2Fjpeg&Expires=1720774229&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyMDc3NDIyOX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9lMGZiZjgwZDcyMmMzMWNhMWMyNjY5NmNmMWQzYjc1ZGI2NTExZjc5ZmNkMmYzZGYxZWMzM2JlMWVkNTI0MDExP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=DYCDxBLdCBs82UYzEnt2Vq7Pkb5soA6N9uHa3FvX2CZlZ1ADFIGbXk8l0TyZpLOtEmhenNqxHdyq4-gYEpz03vgJfu%7EpgInrwHVVrUmFb5DHaf9pv538YMrTTGUfoE7FqoG4IckuOEw8g5MvJbVqz7b4h7A1MGT7N2Zn4JoTj7s%7EyYikzpz2nwA9JhSa8EbW7U846DP-nXJubjyyWJ96alPOnHQkTzwrr4%7E3lTdqc6OtyJzQVBZnkijOV5P7OHx8zvtUHNKkoE-MEflAqVsbOv7HvRYyGPab21DhOScOcS-1VhLyMwEwW2VVDMFEHn44Zi5hMzUg0SAtNAn1604EIw__&Key-Pair-Id=K3ESJI6DHPFC7",
    "This is an image of ",
]

inputs = processor(prompt, return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])

#%%

prompt = ["User:",
           "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg",
           "Describe this image.\nAssistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building.\n",
           "User:",
           "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
           "Describe this image.\nAssistant:"
           ]

inputs = processor(prompt, return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])
# %%
from icecream import ic
import torch

def bytes_to_giga_bytes(n_bytes):
  return n_bytes / 1024 / 1024 / 1024

ic(bytes_to_giga_bytes(torch.cuda.max_memory_allocated("cuda:0")))
ic(bytes_to_giga_bytes(torch.cuda.max_memory_allocated("cuda:1")))
None
# %%
import gc

del model
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats("cuda:0")
  torch.cuda.reset_peak_memory_stats("cuda:1")
  
flush()
# %%
