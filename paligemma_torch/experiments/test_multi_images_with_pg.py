# %%
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# %%
image_files = [
    "/data/images/image2.jpg",
    "/data/images/image1.jpg",
]  # Example image file paths
raw_images = [Image.open(image_file) for image_file in image_files]

# %%
# prompt = "Explain the difference between 2 images"
prompt = "Explain the images"
inputs = processor(prompt, raw_images, return_tensors="pt").to("cuda:1")
output = model.generate(**inputs, max_new_tokens=256)
result = processor.decode(output[0], skip_special_tokens=True)[len(prompt) :]
print(result)

# %%
