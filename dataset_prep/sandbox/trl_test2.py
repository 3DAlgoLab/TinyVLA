from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("imdb", split="train")

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,  # type: ignore
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()
