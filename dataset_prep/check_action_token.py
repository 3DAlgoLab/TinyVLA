import token

from transformers import AutoTokenizer

# model_name = "bczhou/TinyLLaVA-3.1B"
# model_name = "bert-base-uncased"
# model_name = "google/gemma-2b"
model_name = "google/paligemma-3b-pt-224"
tokenizer = AutoTokenizer.from_pretrained(model_name)

for i in range(0, 10000):
    integer_text = f"{i}"
    tokens = tokenizer.tokenize(integer_text)

    if len(tokens) == 1:
        print(f"The integer '{integer_text}' has unique tokens: {tokens[0]}")
    else:
        print(f"The integer '{integer_text}' does not have unique tokens.")
        print("tokens:", tokens)
        break


# another test
tokens = tokenizer.tokenize("foo bar")
print(tokens)
