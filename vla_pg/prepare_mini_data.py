# %% Check Sentence Piece Model(Tokenizer)
import sentencepiece as spm
import os
from icecream import ic
import numpy as np
import tensorflow_datasets as tfds

# %%

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


def check_action_tokens():
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
    return int((action_val * 1000.0 + MAX_STEP_RANGE / 2) * 255 / 60 + 0.5) + START_ID


def action_id_to_trans_val(action_id):
    action_id = np.clip(action_id, 0, 255)
    return (action_id - 128) * MAX_STEP_RANGE / 255 / 1000.0


# ic(action_id_to_trans_val(128))
# ic(action_id_to_trans_val(256))
# ic(action_id_to_trans_val(0))

# ic(trans_val_to_id(0))
# ic(trans_val_to_id(0.030))
# ic(trans_val_to_id(-0.030))


# %%
DATASET_PATH = "/data/language_table_sim"


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def generate_jsonl_data(
    episode_num=50, out_data_folder="./data", dataset_path=DATASET_PATH
):
    builder = tfds.builder_from_directory(dataset_path)
    episode_ds = builder.as_dataset(split="train")

    # print(episode_ds.element_spec)
    episodes_iter = iter(episode_ds.take(episode_num))
    frames = []
    actions = []
    instructions = []
    for episode in episodes_iter:
        for step in episode["steps"].as_numpy_iterator():
            frames.append(step["observation"]["rgb"])
            actions.append(step["action"])
            instructions.append(decode_inst(step["observation"]["instruction"]))

    print("Total Frames:", len(frames))
    return instructions, frames, actions


# %%


if __name__ == "__main__":
    # download_tokenizer()
    # check_action_tokens()
    instructions, frames, actions = generate_jsonl_data()
    ic(len(instructions))
    ic(instructions[:5])
    ic(actions[:5])
