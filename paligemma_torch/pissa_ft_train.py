# %%
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(
    # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
    init_lora_weights="pissa_niter_4",  # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
)
peft_model = get_peft_model(model, lora_config)

peft_model.print_trainable_parameters()

# %%

dataset = load_dataset("imdb", split="train[:1%]")

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=128,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("pissa-llama-2-7b")

# %%
