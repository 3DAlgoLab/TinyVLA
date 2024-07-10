# %%
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

prompt = "Somatic hypermutation allows the immune system to"
generator = pipeline(
    "text-generation", model="3dalgo/test_awesome_eli5_clm-model", tokenizer=tokenizer
)
generator(prompt)
# %%

from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "Somatic hypermutation allows the immune system to"
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
inputs = tokenizer(prompt, return_tensors="pt").input_ids

# %%
inputs = inputs.to("cuda")
model = AutoModelForCausalLM.from_pretrained("3dalgo/test_awesome_eli5_clm-model").to(
    "cuda"
)
outputs = model.generate(
    inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95
)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
# %%
