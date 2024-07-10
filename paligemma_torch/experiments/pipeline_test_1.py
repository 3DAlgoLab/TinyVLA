#%%
from transformers import pipeline

vqa = pipeline(model="impira/layoutlm-document-qa", device=0)
output = vqa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?",
)
output[0]["score"] = round(output[0]["score"], 3)
output
# %%
