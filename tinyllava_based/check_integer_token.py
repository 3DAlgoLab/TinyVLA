from transformers import AutoTokenizer

# hf_path = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
hf_path = "microsoft/phi-2"
# model_name = "bczhou/TinyLLaVA-3.1B"
# model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(hf_path)

print("--- Model or path name:", hf_path)
not_display_unique_token = False
for i in range(0, 1_000):
    integer_text = f"{i}"
    tokens = tokenizer.tokenize(integer_text)

    if len(tokens) == 1:
        if not not_display_unique_token:
            print(f"The integer '{integer_text}' has unique tokens: {tokens[0]}")
    else:
        print(f"The integer '{integer_text}' does not have unique tokens.")
        print("Bye bye...")
        break
