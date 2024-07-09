# %%
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from icecream import ic

# %%

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
# Performs SVD again to initialize the residual model and loads the state_dict of the fine-tuned PiSSA modules.
model = PeftModel.from_pretrained(base_model, "./pissa-llama-2-7b")
model = model.merge_and_unload()

# %%

dataset = load_dataset("imdb", split="test[:1%]")
ic(len(dataset))

# %%
base_model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
device = "cuda"
for item in dataset:
    print("-" * 80)
    print(item["text"])
    input_length = len(item["text"])
    print("* Input Length:", input_length)
    
    inputs = tokenizer([item["text"]], return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    # need to check other result 
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("-" * 80)
    print(result[0][input_length:])
    # print("last token:", result[0][-1])
    # print("last token(bytes):", result[0][-1].encode("utf-8"))
    print("* Output Length:", len(result[0]))
    break
# %%
for index, item in enumerate(dataset):    
    if index == 249:
        print(item)
        break 
    pass
# %%
