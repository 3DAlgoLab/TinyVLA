# %%
import base64
import html
import io

import numpy as np
import supervision as sv
from IPython.display import HTML, display
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# %%

peft_model_id = "3dalgo/language_table_mini"
base_model_id = "google/paligemma-3b-pt-224"


def get_general_lora_model():
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        peft_model_id, device_map="auto"
    )
    return model


def get_merged_model():
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.merge_and_unload()
    return model


# model = get_general_lora_model()
model = get_merged_model()
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")


# %%


def render_inline(image, resize=(128, 128)):
    """Convert image into inline html."""
    image = Image.fromarray(image)
    image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format="jpeg")
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"


def render_example(instruction, image, caption):
    # image = ((image + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
    image = np.asarray(image).copy()  # important!
    h, w, _ = image.shape
    caption = instruction + " " + caption
    caption = caption.strip()
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:128px; height:128px;" src="{render_inline(image, resize=(128,128))}" />
    <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
</div>
"""


# %%
# Check performance of object detection
from datasets import Image as dsImage
from datasets import load_dataset

parent_folder = "/data/language_table_mini_train/"
valid_file = parent_folder + "_annotations.jsonl"


def add_abs_path(example):
    example["image"] = parent_folder + example["image"]
    return example


ds_lt = load_dataset("json", data_files={"valid": valid_file})

ds_lt = ds_lt.map(add_abs_path)
ds_lt = ds_lt.cast_column("image", dsImage(mode="RGB"))
valid_ds = ds_lt["valid"]
valid_ds[0]

# %%
import time

start_time = time.time()
targets = []
predictions = []
outs = []

candidates = np.random.randint(0, len(valid_ds), 10)

# %%
for i, item in enumerate(candidates):
    item = valid_ds[int(candidates[i])]

    raw_image = item["image"]
    w, h = raw_image.size

    prompt = item["prefix"]
    inputs = processor(prompt, raw_image.convert("RGB"), return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt) :]
    result = result.strip()
    outs.append(render_example(prompt, raw_image, result))


display(HTML("".join(outs)))

# %%
