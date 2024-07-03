# Define the mesh
import functools
import io
import math
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import big_vision.utils
import jax
import jax.numpy as jnp
import jax.random as rand
import kagglehub
import ml_collections
import numpy as np
import optax
import sentencepiece
import tensorflow as tf
import tree
from big_vision.datasets import jsonl
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
from PIL import Image
from rich.progress import Progress

import util

# device setup
tf.config.set_visible_devices([], "GPU")

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
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LR: float = 0.005
    BATCH_SIZE: int = 16
    N_ACCUMULATION_STEPS: int = 1
    MAX_SEQ_LEN: int = 128
    N_EPOCHS: int = 1
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
        # l - num index for 18 layers
        if "kv_einsum" in path:
            v_lora_A = lora_params[path]["v_lora_A"]
            v_lora_B = lora_params[path]["v_lora_B"]

            merged_V = v[:, 1:2] + jnp.einsum(
                "ldhmr, ldhrk -> ldhmk", v_lora_A, v_lora_B
            )  # this gives us the same dimension as v[1]

            return jnp.concatenate([v[:, 0:1], merged_V], axis=1)

        elif "q_einsum" in path:
            return v + jnp.einsum(
                "lhmr,lhrv->lhmv",
                lora_params[path]["q_lora_A"],
                lora_params[path]["q_lora_B"],
            )
        elif "qkv_einsum" in path:
            # WARNING: Assume the first axis would be layers(n_layers), presented as 'l'
            q_ = v[:, 0:1]
            k_ = v[:, 1:2]
            v_ = v[:, 2:3]
            q_lora_A = lora_params[path]["q_lora_A"]
            q_lora_B = lora_params[path]["q_lora_B"]
            v_lora_A = lora_params[path]["v_lora_A"]
            v_lora_B = lora_params[path]["v_lora_B"]

            merged_q = q_ + jnp.einsum("lhmr,lhrk->lhmk", q_lora_A, q_lora_B)
            merged_v = v_ + jnp.einsum("lhmr,lhrk->lhmk", v_lora_A, v_lora_B)

            return jnp.concatenate([merged_q, k_, merged_v], axis=1)
        else:
            return v

    merged_params = tree.map_structure_with_path(merge_fn, params)
    return merged_params


def preprocess_image(image, size=224):
    """Model has been trained to handle images of different aspects ratios
    resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
    options are helpful to improve quality in some tasks."""

    image = np.asarray(image)
    if image.ndim == 2:  # Convert image without last channel into greyscale.
        image = np.stack((image,) * 3, axis=-1)
    image = image[..., :3]  # Remove alpha layer.
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method="bilinear", antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]


def preprocess_tokens(tokenizer, prefix, suffix=None, seqlen=None):
    """Model has been trained to handle tokenized text composed of a prefix with
    full attention and a suffix with causal attention."""
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)  # type: ignore
    mask_ar = [0] * len(tokens)  # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)  # type: ignore
        tokens += suffix
        mask_ar += [1] * len(suffix)  # 1 to use causal attention for suffix.
        mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

    mask_input = [1] * len(tokens)  # 1 if it's a token, 0 if padding.
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))


def postprocess_tokens(tokenizer, tokens):
    tokens = tokens.tolist()  # np.array to list[int]
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)  # type: ignore


def gen_data_iterator(tf_dataset, tokenizer, train=False, seq_length=128):
    dataset = tf_dataset
    for example in dataset.as_numpy_iterator():
        image = Image.open(io.BytesIO(example["image"]))  # type: ignore
        image = preprocess_image(image)

        # prefix = "caption en"  # Could also be a different prefix per example.

        prefix = example["prefix"].decode().lower()  # type: ignore
        suffix = example["suffix"].decode().lower()  # type: ignore

        if train:
            tokens, mask_ar, mask_loss, _ = preprocess_tokens(
                tokenizer, prefix, suffix, seq_length
            )
            label, _, _, _ = preprocess_tokens(tokenizer, suffix, seqlen=seq_length)

            yield {
                "image": np.asarray(image),
                "text": np.asarray(tokens),
                "label": np.asarray(label),
                "mask_ar": np.asarray(mask_ar),
                "mask_loss": np.asarray(mask_loss),
            }

        else:
            tokens, mask_ar, _, mask_input = preprocess_tokens(
                tokenizer, prefix, seqlen=seq_length
            )
            label, _, _, _ = preprocess_tokens(tokenizer, suffix, seqlen=seq_length)

            yield {
                "image": np.asarray(image),
                "text": np.asarray(tokens),
                "label": np.asarray(label),
                "mask_ar": np.asarray(mask_ar),
                "mask_input": np.asarray(mask_input),
            }


def train_lora(
    config: PaliGemmaConfig,
    checkpoint_dir: str,
    train_dataset,
    val_dataset,
    verbose=True,
    checkpoint_prefix="lora",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    mesh = jax.sharding.Mesh(jax.devices(), ("data"))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    download_tokenizer(config.TOKENIZER_PATH)
    tokenizer = sentencepiece.SentencePieceProcessor(config.TOKENIZER_PATH)  # type: ignore

    key = rand.PRNGKey(config.SEED)
    with jax.default_device(cpu_device):
        ckpt_path = download_model(config.MODEL_PATH)
        model, params = load_model(ckpt_path)

    # Define `decode` function to sample outputs from the model.
    decode_fn = predict_fns.get_all(model)["decode"]
    decode = functools.partial(
        decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id()
    )

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
            value_shape = v[0][1].shape

            # (n_layers, 1, n_heads, d_m, r) x  (n_layers, 1, n_heads, r, d_v)
            v_A_shape = (v.shape[0], 1, *value_shape[:-1], lora_r)
            v_B_shape = (
                v.shape[0],
                1,
                value_shape[0],  # n_heads
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

    util.params_to_json(lora_map, "lora_params.json")
    if is_process_0 and verbose:
        print("Successfully loaded and sharded model parameters!")

    # prepare train data & validation iterator
    num_batches = train_dataset.total_examples // config.BATCH_SIZE
    train_dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
    val_dataset = val_dataset.get_tfdata(ordered=True)

    train_data_iter = gen_data_iterator(train_dataset, tokenizer, train=True)
    val_data_iter = gen_data_iterator(val_dataset, tokenizer)

    n_steps = math.ceil(num_batches / config.N_ACCUMULATION_STEPS)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.LR,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=config.LR,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, config.N_ACCUMULATION_STEPS)
    opt_state = optimizer.init(lora_map)

    # verbosity_freq = num_batches // 100
    verbosity_freq = 10
    validation_freq = verbosity_freq * 10
    jax.clear_caches()

    # Training Loop
    n_train_steps = num_batches

    @functools.partial(jax.jit, donate_argnums=(0,))
    def update_fn(lora_params, params, opt_state, batch):
        imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

        def loss_fn(lora_params, pretrained_params):
            params_merged = merge_lora_params(pretrained_params, lora_params)

            text_logits, _ = model.apply(
                {"params": params_merged},  # type: ignore
                imgs,
                txts[:, :-1],
                mask_ar[:, :-1],
                train=True,
            )
            logp = jax.nn.log_softmax(text_logits, axis=-1)

            # The model takes as input txts[:, :-1] but the loss is defined as predicting
            # next tokens txts[:, 1:]. Additionally, mask_loss[:, 1:] indicates which tokens
            # are part of the loss (e.g. prefix and padded tokens are not included).
            mask_loss = batch["mask_loss"][:, 1:]
            targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

            # Compute the loss per example. i.e. the mean of per token pplx.
            # Since each example has a different number of tokens we normalize it.
            token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
            example_loss = -jnp.sum(
                token_pplx * mask_loss, axis=-1
            )  # sum across seq_len.
            example_loss /= jnp.clip(
                jnp.sum(mask_loss, -1), 1
            )  # weight by num of tokens.

            # batch_loss: mean of per example loss.
            return jnp.mean(example_loss)

        loss_value, grads = jax.value_and_grad(loss_fn)(lora_params, params)
        updates, opt_state = optimizer.update(grads, opt_state, lora_params)

        lora_params = optax.apply_updates(lora_params, updates)
        return lora_params, opt_state, loss_value

    print(f"Model predictions before training...")
    html_out = ""
    for image, _, caption in make_predictions(
        val_data_iter,
        decode,
        params,
        tokenizer,
        data_sharding=data_sharding,
        num_examples=4,
        batch_size=4,
    ):
        html_out += util.render_example(image, caption)
    util.display(util.HTML(html_out))

    print("Training started...!")
    for epoch in range(1, config.N_EPOCHS + 1):
        for step in range(1, n_train_steps + 1):
            # Make list of N training examples.
            examples = [next(train_data_iter) for _ in range(config.BATCH_SIZE)]

            # Convert list of examples into a dict of np.arrays and load onto devices.
            batch = jax.tree.map(lambda *x: np.stack(x), *examples)
            batch = big_vision.utils.reshard(batch, data_sharding)

            # Training step and report training loss
            lora_map, opt_state, loss = update_fn(lora_map, params, opt_state, batch)
            loss = jax.device_get(loss)

            if step % verbosity_freq == 0 and verbose:
                print(f"Epoch:{epoch:02} steps:{step:03} loss:{loss:.3f}")

            if (step % validation_freq) == 0:
                print(f"Model predictions at epoch:{epoch} step:{step}")
                html_out = ""
                params_merged = merge_lora_params(params, lora_map)
                for image, _, caption in make_predictions(
                    val_data_iter,
                    decode,
                    params_merged,
                    tokenizer,
                    data_sharding=data_sharding,
                    num_examples=4,
                    batch_size=4,
                ):
                    html_out += util.render_example(image, caption)
                util.display(util.HTML(html_out))

        # save lora params for this epoch
        with open(
            os.path.join(checkpoint_dir, f"{checkpoint_prefix}_epoch_{epoch}.pickle"),
            "wb",
        ) as f:
            pickle.dump(lora_map, f)

    with open(
        os.path.join(checkpoint_dir, f"{checkpoint_prefix}_final.pickle"), "wb"
    ) as f:
        pickle.dump(lora_map, f)


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


# Evaluation/inference loop.
def make_predictions(
    data_iterator,
    decode,  # Sampler function from big-vision
    params,  # merged params
    tokenizer,
    *,
    data_sharding=None,
    num_examples=None,
    batch_size=4,
    seqlen=128,
    sampler="greedy",
):
    outputs = []
    while True:
        # Construct a list of examples in the batch.
        examples = []
        try:
            for _ in range(batch_size):
                examples.append(next(data_iterator))
                examples[-1]["_mask"] = np.array(True)  # Indicates true example.
        except StopIteration:
            if len(examples) == 0:
                return outputs

        # Not enough examples to complete a batch. Pad by repeating last example.
        while len(examples) % batch_size:
            examples.append(dict(examples[-1]))
            examples[-1]["_mask"] = np.array(False)  # Indicates padding example.

        # Convert list of examples into a dict of np.arrays and load onto devices.
        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)

        # Make model predictions
        tokens = decode(
            {"params": params}, batch=batch, max_decode_len=seqlen, sampler=sampler
        )

        # Fetch model predictions to device and detokenize.
        tokens, mask = jax.device_get((tokens, batch["_mask"]))
        tokens = tokens[mask]  # remove padding examples.
        labels = [postprocess_tokens(tokenizer, e["label"]) for e in examples]
        responses = [postprocess_tokens(tokenizer, t) for t in tokens]

        # Append to html output.
        for example, label, response in zip(examples, labels, responses):
            outputs.append((example["image"], label, response))
            if num_examples and len(outputs) >= num_examples:
                return outputs


if __name__ == "__main__":
    config = PaliGemmaConfig()
    train_ds, val_ds = prepare_test_dataset()
    train_lora(config, "checkpoints", train_ds, val_ds)
