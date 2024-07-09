# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1", padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)

model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(
    "cuda"
)

print(model_inputs)

# %%

generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# %%
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
model_inputs = tokenizer(
    ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
).to("cuda")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# %%
