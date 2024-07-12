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

peft_model_id = "3dalgo/number_op_detection"
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

image_file = (
    "/data/number-ops-1/dataset/40524_jpg.rf.ed19c89e70b5d514078175127cf215fd.jpg"
)
prompt = (
    "detect 0 ; 1 ; 2 ; 3 ; 4 ; 5 ; 6 ; 7 ; 8 ; 9 ; div ; eqv ; minus ; mult ; plus"
)
raw_image = Image.open(image_file)
raw_image

# image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
# %%
inputs = processor(prompt, raw_image.convert("RGB"), return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=20)
result = processor.decode(output[0], skip_special_tokens=True)[len(prompt) :]


# %%

classes = prompt.replace("detect ", "").split(" ; ")


def render_inline(image, resize=(128, 128)):
    """Convert image into inline html."""
    image = Image.fromarray(image)
    image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format="jpeg")
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"


def render_example(image, caption, classes=classes):
    # image = ((image + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
    image = np.asarray(image).copy()  # important!
    h, w, _ = image.shape
    caption = caption.strip()
    try:
        detections = sv.Detections.from_lmm(
            lmm="paligemma", result=caption, resolution_wh=(w, h), classes=classes
        )
        image = sv.BoundingBoxAnnotator().annotate(image, detections)
        image = sv.LabelAnnotator().annotate(image, detections)
    except Exception as e:
        print(caption)
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:128px; height:128px;" src="{render_inline(image, resize=(64,64))}" />
    <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
</div>
"""


display(HTML(render_example(raw_image, result)))

# %%
# Check performance of object detection
from datasets import Image as hfImage
from datasets import load_dataset

parent_folder = "/data/number-ops-1/dataset/"
valid_file = parent_folder + "_annotations.valid.jsonl"


def add_abs_path(example):
    example["image"] = parent_folder + example["image"]
    return example


ds_numbers = load_dataset("json", data_files={"valid": valid_file})

ds_numbers = ds_numbers.map(add_abs_path)
ds_numbers = ds_numbers.cast_column("image", hfImage(mode="RGB"))
valid_ds = ds_numbers["valid"]

# %%
import time

start_time = time.time()
targets = []
predictions = []

for i, item in enumerate(valid_ds):
    raw_image = item["image"]
    w, h = raw_image.size

    if i % 10 == 0:
        print(i)

    if i > 100:
        break

    prompt = item["prefix"]
    inputs = processor(prompt, raw_image.convert("RGB"), return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt) :]
    result = result.strip()

    classes = prompt.replace("detect ", "").split(" ; ")

    target = sv.Detections.from_lmm(
        lmm="paligemma", result=item["suffix"], resolution_wh=(w, h), classes=classes
    )

    targets.append(target)
    prediction = sv.Detections.from_lmm(
        lmm="paligemma", result=result, resolution_wh=(w, h), classes=classes
    )

    prediction.confidence = np.ones(len(prediction))
    predictions.append(prediction)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")

mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)

print(f"map50_95: {mean_average_precision.map50_95:.2f}")
print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")

# @title Calculate Confusion Matrix
confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions, targets=targets, classes=classes
)

_ = confusion_matrix.plot()


# %%
