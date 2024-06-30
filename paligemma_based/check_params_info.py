import json
import os

import big_vision.utils
import jax
import kagglehub
import ml_collections
from big_vision.models.proj.paligemma import paligemma


def parameter_overview(params, file_name="parameters_paligemma.txt"):
    with open(file_name, "wt") as f:
        for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
            f.write(f"{path} {str(arr.shape)} {arr.dtype}\n")
            # f.write(f"{path:80s} {str(arr.shape):22s} {arr.dtype}\n")


def check_tree1(params):
    # result = big_vision.utils.tree_flatten_with_names(params)

    result = jax.tree.map(lambda x: x.shape, params)
    print(result)
    # breakpoint()
    with open("params_structure.json", "wt") as f:
        json.dump(result, f)


def check_pretrained_model():
    MODEL_PATH = "./pt_224_128.params.f16.npz"
    if not os.path.exists(MODEL_PATH):
        print(
            "Downloading the checkpoint from Kaggle, this could take a few minutes...."
        )
        # Download only the float16 model.
        MODEL_PATH = kagglehub.model_download(
            "google/paligemma/jax/paligemma-3b-pt-224", "paligemma-3b-pt-224.f16.npz"
        )
        print(f"Model path: {MODEL_PATH}")

    TOKENIZER_PATH = "./paligemma_tokenizer.model"
    if not os.path.exists(TOKENIZER_PATH):
        print("Downloading the model tokenizer...")
        os.system(
            f"gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}"
        )
        print(f"Tokenizer path: {TOKENIZER_PATH}")

    model_config = ml_collections.FrozenConfigDict(
        {
            "llm": {"vocab_size": 257_152},
            "img": {
                "variant": "So400m/14",
                "pool_type": "none",
                "scan": True,
                "dtype_mm": "float16",
            },
        }
    )
    model = paligemma.Model(**model_config)  # type: ignore
    print(model)

    params = paligemma.load(None, MODEL_PATH, model_config)
    # parameter_overview(params)
    check_tree1(params)


if __name__ == "__main__":
    check_pretrained_model()
