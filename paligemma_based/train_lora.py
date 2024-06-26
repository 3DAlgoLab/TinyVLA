# Define the mesh
import math
import os
from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as rand
import kagglehub
import ml_collections
import optax
import sentencepiece
import tree
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
from rich.progress import Progress
from tqdm import tqdm

# device setup
cpu_devices = jax.devices("cpu")
gpu_devices = jax.devices("gpu")
gpu_sharding_mp = PositionalSharding(gpu_devices)
gpu_sharding_mp = gpu_sharding_mp.reshape((1, len(gpu_devices)))
is_process_0 = jax.process_index() == 0
cpu_device = jax.devices("cpu")[0]
gpu_device = jax.devices("gpu")[0]


@dataclass
class PaliGemmaConfig:
    MODEL_PATH: str = "./pt_224_128.params.f16.npz"
    LORA_R: int = 16
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.05
    LR: float = 0.002
    BATCH_SIZE: int = 8
    N_ACCUMULATION_STEPS: int = 2
    MAX_SEQ_LEN: int = 128
    N_EPOCHS: int = 7
    SEED: int = 420
    CACHE_SIZE: int = 30  # number of steps in the transformer's cache ?
    TOKENIZER_PATH: str = "./paligemma_tokenizer.model"


def get_transformer_config_from_params(params, cache_size: int = 1024) -> dict:
    """Creates a TransformerConfig from loaded parameters."""

    num_layers, hidden_dim, embed_dim = params["llm"]["layers"]["mlp"]["linear"].shape
    _, num_heads, head_dim, _ = params["llm"]["layers"]["attn"]["attn_vec_einsum"][
        "w"
    ].shape

    use_qkv_einsum = "qkv_einsum" in params["llm"]["layers"]["attn"]
    if use_qkv_einsum:
        num_kv_heads = num_heads
    else:
        num_kv_heads = params["llm"]["layers"]["attn"]["kv_einsum"]["w"].shape[2]

    num_embed = params["llm"]["embedder"]["input_embedding"].shape[0]
    return dict(
        num_layers=num_layers,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        max_cache_length=cache_size,
    )


def download_model(path: str):
    if not os.path.exists(path):
        print(
            "Downloading the checkpoint from Kaggle, this could take a few minutes...."
        )
        # Download only the float16 model.
        path = kagglehub.model_download(
            "google/paligemma/jax/paligemma-3b-pt-224", "paligemma-3b-pt-224.f16.npz"
        )
        print(f"Model path: {path}")

    return path


def download_tokenizer(path: str):
    if not os.path.exists(path):
        print("Downloading the model tokenizer...")
        os.system(f"gsutil cp gs://big_vision/paligemma_tokenizer.model {path}")
        print(f"Tokenizer path: {path}")

    return path


def load_model(ckpt_path: str):
    model_args = ml_collections.FrozenConfigDict(
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
    model = paligemma.Model(**model_args)  # type: ignore
    # Load params - this can take up to 1 minute in T4 colabs.
    params = paligemma.load(None, ckpt_path, model_args)

    return model, params


def merge_lora_params(params, lora_params: Dict):
    def merge_fn(path, v):
        # h - num heads, m - model dim, r - lora dim, k - key dim, v - value dim
        if "kv_einsum" in path:
            v_lora_A = lora_params[path]["v_lora_A"]
            v_lora_B = lora_params[path]["v_lora_B"]
            merged_V = v[1] + jnp.einsum(
                "hmr,hrk->hmk", v_lora_A, v_lora_B
            )  # this gives us the same dimension as v[1]
            return jnp.stack([v[0], merged_V])
        elif "q_einsum" in path:
            return v + jnp.einsum(
                "hmr,hrv->hmv",
                lora_params[path]["q_lora_A"],
                lora_params[path]["q_lora_B"],
            )
        elif "qkv_einsum" in path:
            q_ = v[0]
            k_ = v[1]
            v_ = v[2]
            q_lora_A = lora_params[path]["q_lora_A"]
            q_lora_B = lora_params[path]["q_lora_B"]
            v_lora_A = lora_params[path]["v_lora_A"]
            v_lora_B = lora_params[path]["v_lora_B"]

            merged_q = q_ + jnp.einsum("hmr,hrk->hmk", q_lora_A, q_lora_B)
            merged_v = v_ + jnp.einsum("hmr,hrk->hmk", v_lora_A, v_lora_B)

            return jnp.stack([merged_q, k_, merged_v])
        else:
            return v

    merged_params = tree.map_structure_with_path(merge_fn, params)
    return merged_params


def train_lora(config: PaliGemmaConfig, train_dataset, checkpoint_dir, verbose=True):

    global optimize
    download_tokenizer(config.TOKENIZER_PATH)
    tokenizer = sentencepiece.SentencePieceProcessor(config.TOKENIZER_PATH)  # type: ignore
    dataset = train_dataset
    key = rand.PRNGKey(config.SEED)

    with jax.default_device(cpu_device):
        ckpt_path = download_model(config.MODEL_PATH)
        model, params = load_model(ckpt_path)

    model_config = get_transformer_config_from_params(params)
    num_gpus = jax.device_count("gpu")
    devices = mesh_utils.create_device_mesh((num_gpus,))
    default_mesh = Mesh(devices, axis_names=("p",))

    def mesh_sharding(p_spec: P, mesh=None) -> NamedSharding:
        if mesh is None:
            mesh = default_mesh
        return NamedSharding(mesh, p_spec)

    lora_map = {}
    lora_r = config.LORA_R

    def param_shard_func(path, v):
        if "input_embedding" in path:
            return jax.device_put(
                v,
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
        elif "attn_vec_einsum" in path:
            return jax.device_put(
                v,
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
        elif "kv_einsum" in path:
            # get the key
            v0 = v[0]  # first layer
            value_shape = (v.shape[0], *v0[1].shape)  # (n_layers, n_heads, d_m, d_v)

            # (n_layers, n_heads, d_m, r) x  (n_layers, n_heads, r, d_v)
            v_A_shape = (*value_shape[:-1], lora_r)
            v_B_shape = (
                value_shape[0],  # n_layers
                value_shape[1],  # n_heads
                lora_r,
                value_shape[-1],  # d_v
            )
            # create the lora params
            with jax.default_device(cpu_device):
                v_lora_A = jnp.zeros(v_A_shape, dtype=jnp.bfloat16)
                v_lora_B = jnp.zeros(v_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {"v_lora_A": v_lora_A, "v_lora_B": v_lora_B}

            return (
                jax.device_put(
                    v,
                    mesh_sharding(
                        P(
                            None,
                        )
                    ),
                )
                if v.shape[2] == 1
                else jax.device_put(
                    v,
                    mesh_sharding(
                        P(
                            None,
                            None,
                            "p",
                        )
                    ),
                )
            )
        elif "q_einsum" in path:
            q_A_shape = (*v.shape[:-1], lora_r)  # (n_layers, n_heads, d_m, r)
            q_B_shape = (
                *v.shape[:2],
                lora_r,
                v.shape[-1],
            )  # (n_layers, n_heads, r, d_k)
            # create the lora params
            with jax.default_device(cpu_device):
                q_lora_A = jnp.zeros(q_A_shape, dtype=jnp.bfloat16)
                q_lora_B = jnp.zeros(q_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {"q_lora_A": q_lora_A, "q_lora_B": q_lora_B}

            return jax.device_put(v, mesh_sharding(P("p", None, None)))
        elif "qkv_einsum" in path:
            # WARNING: Assume the first axis would be layers(n_layers)
            # v.shape (n_layers, 3, n_heads, d_m, d_k)
            value_shape = (v.shape[0], *v[0][1].shape)  # (n_layers, n_heads, d_m, d_v)
            v_A_shape = (*value_shape[:-1], lora_r)  # (n_heads, d_m, r)
            v_B_shape = (
                value_shape[0],
                value_shape[1],
                lora_r,
                value_shape[-1],
            )  # (n_heads, r, d_k)

            q_A_shape, q_B_shape = v_A_shape, v_B_shape

            with jax.default_device(cpu_device):
                v_lora_A = jnp.zeros(v_A_shape, dtype=jnp.bfloat16)
                v_lora_B = jnp.zeros(v_B_shape, dtype=jnp.bfloat16)
                q_lora_A = jnp.zeros(q_A_shape, dtype=jnp.bfloat16)
                q_lora_B = jnp.zeros(q_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {
                "v_lora_A": v_lora_A,
                "v_lora_B": v_lora_B,
                "q_lora_A": q_lora_A,
                "q_lora_B": q_lora_B,
            }
            return jax.device_put(
                v,
                mesh_sharding(
                    P(
                        None,
                        None,
                        "p",
                    )
                ),
            )
        elif "gating_einsum" in path:
            return jax.device_put(v, mesh_sharding(P(None, None, None, "p")))
        elif "linear" in path:
            return jax.device_put(
                v,
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
        else:
            # replicate across all gpus
            return jax.device_put(v, mesh_sharding(P(*((None,) * len(v.shape)))))

    params = tree.map_structure_with_path(param_shard_func, params)

    # initialize the lora A's to be sampled from a normal distribution
    for k, v in lora_map.items():
        for k1, v1 in v.items():
            if "lora_A" in k1:
                with jax.default_device(cpu_device):
                    key, split = rand.split(key, 2)
                    v[k1] = rand.normal(split, v1.shape, dtype=jnp.bfloat16)

    # shard the lora params
    for k, v in lora_map.items():
        if "kv_einsum" in k:
            # check if only 1 kv head
            if model_config["num_kv_heads"] == 1:
                # no sharding
                v["v_lora_A"] = jax.device_put(
                    v["v_lora_A"],
                    mesh_sharding(
                        P(
                            None,
                        )
                    ),
                )
                v["v_lora_B"] = jax.device_put(
                    v["v_lora_B"],
                    mesh_sharding(
                        P(
                            None,
                        )
                    ),
                )
            else:
                v["v_lora_A"] = jax.device_put(
                    v["v_lora_A"],
                    mesh_sharding(
                        P(
                            "p",
                        )
                    ),
                )
                v["v_lora_B"] = jax.device_put(
                    v["v_lora_B"],
                    mesh_sharding(
                        P(
                            "p",
                        )
                    ),
                )
        elif "q_einsum" in k:
            v["q_lora_A"] = jax.device_put(
                v["q_lora_A"],
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
            v["q_lora_B"] = jax.device_put(
                v["q_lora_B"],
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
        elif "qkv_einsum" in k:
            # both q and v are going to be sharded
            v["q_lora_A"] = jax.device_put(
                v["q_lora_A"],
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
            v["q_lora_B"] = jax.device_put(
                v["q_lora_B"],
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
            v["v_lora_A"] = jax.device_put(
                v["v_lora_A"],
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )
            v["v_lora_B"] = jax.device_put(
                v["v_lora_B"],
                mesh_sharding(
                    P(
                        "p",
                    )
                ),
            )

    if is_process_0 and verbose:
        print("Successfully loaded and sharded model parameters!")

    n_steps = math.ceil(len(dataloader) / config.N_ACCUMULATION_STEPS)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.LR,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=config.LR,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, config.N_ACCUMULATION_STEPS)
    optimize = optimizer.update

    # opt_state = optimizer.init(params)
    opt_state = optimizer.init(lora_map)

    num_batches = len(dataloader)
    verbosity_freq = num_batches // 100
    jax.clear_caches()


if __name__ == "__main__":
    config = PaliGemmaConfig()
    train_lora(config, None, None)
