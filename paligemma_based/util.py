import base64
import html
import io
import json
import os
import pickle
from typing import List

import cv2
import jax
import numpy as np
import supervision as sv
from big_vision.datasets import jsonl
from IPython.core.display import HTML, display
from PIL import Image
from tqdm.notebook import tqdm


def params_to_json(params, file_name="params_structure.json"):
    result = jax.tree.map(lambda x: x.shape, params)
    # print(result)

    filtered_obj = dict()
    for k, v in result.items():
        if isinstance(k, tuple):
            k = f"{k}"
        filtered_obj[k] = v

    # breakpoint()
    with open(file_name, "wt") as f:
        json.dump(filtered_obj, f)


def json_dump_test():
    with open("dumped.pkl", "rb") as f:
        result = pickle.load(f)

    print(result)
    file_name = "lora_param.json"
    with open(file_name, "wt") as f:
        json.dump(result, f)


def split_and_keep_second_part(s):
    parts = s.split("\n", 1)
    if len(parts) > 1:
        return parts[1]
    return s


def render_inline(image, resize=(128, 128)):
    """Convert image into inline html."""
    image = Image.fromarray(image)
    image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format="jpeg")
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"


def render_example(image, caption, classes=[]):
    assert len(classes) > 0
    image = ((image + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
    h, w, _ = image.shape
    try:
        detections = sv.Detections.from_lmm(
            lmm="paligemma", result=caption, resolution_wh=(w, h), classes=classes
        )
        image = sv.BoundingBoxAnnotator().annotate(image, detections)
        image = sv.LabelAnnotator().annotate(image, detections)
    except:
        print(caption)
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:128px; height:128px;" src="{render_inline(image, resize=(64,64))}" />
    <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
</div>
"""


def read_n_lines(file_path: str, n: int) -> List[str]:
    with open(file_path, "r") as file:
        lines = [next(file).strip() for _ in range(n)]
    return lines


def check_test_data(location):
    images = []
    lines = read_n_lines(f"{location}/dataset/_annotations.train.jsonl", 25)
    first = json.loads(lines[0])

    CLASSES = first.get("prefix").replace("detect ", "").split(" ; ")

    for line in lines:
        data = json.loads(line)
        image = cv2.imread(f"{location}/dataset/{data.get('image')}")
        (h, w, _) = image.shape
        detections = sv.Detections.from_lmm(
            lmm="paligemma",
            result=data.get("suffix"),
            resolution_wh=(w, h),
            classes=CLASSES,
        )

        image = sv.BoundingBoxAnnotator(thickness=4).annotate(image, detections)
        image = sv.LabelAnnotator(text_scale=2, text_thickness=4).annotate(
            image, detections
        )
        images.append(image)

    sv.plot_images_grid(images, (5, 5))

    return CLASSES


def prepare_test_dataset():
    """Roboflow number image dataset

    Returns:
        train_dataset: for training, tf_dataset
        val_dataset: for validation
    """
    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("roboflow-jvuqo").project("number-ops-j1426")
    version = project.version(1)
    dataset = version.download("paligemma")

    train_dataset = jsonl.DataSource(
        os.path.join(dataset.location, "dataset/_annotations.train.jsonl"),
        fopen_keys={"image": f"{dataset.location}/dataset"},
    )

    val_dataset = jsonl.DataSource(
        os.path.join(dataset.location, "dataset/_annotations.valid.jsonl"),
        fopen_keys={"image": f"{dataset.location}/dataset"},
    )

    # train_dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
    # val_dataset = val_dataset.get_tfdata(ordered=True)

    return train_dataset, val_dataset


if __name__ == "__main__":
    json_dump_test()
