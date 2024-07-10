from transformers import AutoTokenizer 
from icecream import ic

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
ic(encoded_input)

ic(tokenizer.decode(encoded_input["input_ids"]))