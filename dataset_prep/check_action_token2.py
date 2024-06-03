from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
example = "This is a tokenization example"
enc = tokenizer(example, add_special_tokens=False)

desired_output = []
for w_idx in set(enc.word_ids()):
    start, end = enc.word_to_tokens(w_idx)
    # Add +1 because you want to start with 1 (not 0)
    start += 1
    end += 1
    desired_output.append(list(range(start, end)))

print(desired_output)  # Output: [[1], [2], [3], [4, 5], [6]]
