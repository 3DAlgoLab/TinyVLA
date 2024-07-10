#%%
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

#%%
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

#%%
prompt = "What is on the flower?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors="pt").to('cuda')
output = model.generate(**inputs, max_new_tokens=20)
output_decoded = processor.decode(output[0], skip_special_tokens=True)
print(output_decoded[len(prompt) :])
print("Finished.")

# %%
