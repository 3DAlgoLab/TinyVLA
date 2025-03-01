# %% Check Sentence Piece Model(Tokenizer)

import json
import os
import shutil
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tensorflow_datasets as tfds
from icecream import ic
from PIL import Image

TOKENIZER_PATH = "./paligemma_tokenizer.model"


def download_tokenizer(tokenizer_path=TOKENIZER_PATH):
    if not os.path.exists(tokenizer_path):
        print("Downloading the model tokenizer...")
        os.system(
            f"gsutil cp gs://big_vision/paligemma_tokenizer.model {tokenizer_path}"
        )
        print(f"Tokenizer path: {tokenizer_path}")
    else:
        print(f"Tokenizer file: {tokenizer_path} is already downloaded")


def check_action_tokens(tokenizer_path=TOKENIZER_PATH):
    sp = spm.SentencePieceProcessor(tokenizer_path)  # type: ignore

    start_id = 256_000
    reserved_size = 256
    for i in range(start_id, start_id + reserved_size):
        piece = sp.IdToPiece(i)
        print(f"id:{i} -->piece:{piece}")


# %% Language Table Simulation based dataset
START_ID = 0
MAX_STEP_RANGE = 60  # mm (-30 ~ + 30)


def trans_val_to_id(action_val):
    action_val = np.clip(action_val, -0.03, 0.03)
    return (
        (action_val * 1000.0 + np.ones_like(action_val) * MAX_STEP_RANGE / 2) * 255 / 60
        + 0.5
    ).astype(int) + START_ID


def action_id_to_trans_val(action_id):
    action_id = np.clip(action_id, 0, 255)
    return (action_id - 128) * MAX_STEP_RANGE / 255 / 1000.0


def test_trans_val_to_id():
    val = action_id_to_trans_val(128)
    ic(val)
    assert trans_val_to_id(val) == 128
    val = action_id_to_trans_val(220)
    ic(val)
    assert trans_val_to_id(val) == 220


def language_table_action_to_rt2_action_string(action):
    """Only consider Terminate, X, Y,
    RT-2 Format: [Terminate, DX, DY, DZ, RX, RY, RZ, Grip]
    """

    # Assume zero action is terminate
    is_terminated = np.allclose(action, np.zeros_like(action))
    terminate_id = 255 if is_terminated else 0
    action_id = trans_val_to_id(action)

    action_ids = [terminate_id, action_id[0], action_id[1], 128, 128, 128, 128, 0]

    return " ".join(map(lambda v: f"<loc{v:04}>", action_ids))


def language_table_action_to_xy_string(action):
    """Only consider Terminate, X, Y,
    No Spaces
    """
    # Assume zero action is terminate
    is_terminated = np.allclose(action, np.zeros_like(action))
    terminate_id = 255 if is_terminated else 0
    action_id = trans_val_to_id(action)
    action_ids = [terminate_id, action_id[0], action_id[1]]
    return "".join(map(lambda v: f"<loc{v:04}>", action_ids))


# %%
DATASET_PATH = "/data/language_table_sim"


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def check_dataset_size(dataset_path=DATASET_PATH):
    builder = tfds.builder_from_directory(dataset_path)
    episode_ds = builder.as_dataset(split="train")
    ic(len(episode_ds))


def generate_jsonl_data(
    episode_num=5,
    out_data_folder="./data/language_table_mini",
    dataset_path=DATASET_PATH,
    episode_skip=None,
    action_to_string=language_table_action_to_rt2_action_string,
):
    builder = tfds.builder_from_directory(dataset_path)
    episode_ds = builder.as_dataset(split="train")

    # print(episode_ds.element_spec)
    if episode_skip is not None:
        episode_ds = episode_ds.skip(episode_skip)
    episodes_iter = iter(episode_ds.take(episode_num))  # type: ignore
    frames = []
    actions = []
    instructions = []
    for episode in episodes_iter:
        for step in episode["steps"].as_numpy_iterator():
            frames.append(step["observation"]["rgb"])
            actions.append(step["action"])
            instructions.append(decode_inst(step["observation"]["instruction"]))

    # ensure out_data_folder exists
    out_data_folder = Path(out_data_folder)
    shutil.rmtree(out_data_folder, ignore_errors=True)
    out_data_folder.mkdir(parents=True, exist_ok=True)
    attrib_filename = Path(out_data_folder) / "_annotations.jsonl"

    instruction_template = "What should the robot do to {task}?"
    with open(attrib_filename, "w") as f:
        print("Total Frames:", len(frames))
        for i, (instruction, frame, action) in enumerate(
            zip(instructions, frames, actions)
        ):
            image_file_name = f"frame_{i:08_}.png"
            frame_path = out_data_folder / image_file_name
            action_str = action_to_string(action)

            prefix_str = instruction_template.format(task=instruction)
            element = dict(prefix=prefix_str, image=image_file_name, suffix=action_str)
            Image.fromarray(frame).save(frame_path)
            f.write(f"{json.dumps(element)}\n")


# %%

if __name__ == "__main__":

    # Test
    # test_trans_val_to_id()
    check_dataset_size()

    #
    generate_jsonl_data(
        2000,
        "/data/language_table_mini_train2",
        action_to_string=language_table_action_to_xy_string,
    )
    generate_jsonl_data(
        5,
        "/data/language_table_mini_valid2",
        episode_skip=2000,
        action_to_string=language_table_action_to_xy_string,
    )
