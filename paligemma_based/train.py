# %%
import base64
import functools
import html
import io
import os
import warnings

import jax
import jax.lib
import jax.numpy as jnp
import kagglehub
import ml_collections
import numpy as np
import sentencepiece
import tensorflow as tf
from IPython.core.display import HTML, display
from PIL import Image

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.sharding
import big_vision.utils

# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

SEQLEN = 128


def init_devices():
    # Don't let TF use the GPU or TPUs
    tf.config.set_visible_devices([], "GPU")
    tf.config.set_visible_devices([], "TPU")

    backend = jax.lib.xla_bridge.get_backend()
    print(f"JAX version:  {jax.__version__}")
    print(f"JAX platform: {backend.platform}")
    print(f"JAX devices:  {jax.device_count()}")


def load_pretrained_model():
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
    tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)  # type: ignore

    # Load params - this can take up to 1 minute in T4 colabs.
    params = paligemma.load(None, MODEL_PATH, model_config)

    # Define `decode` function to sample outputs from the model.
    decode_fn = predict_fns.get_all(model)["decode"]
    decode = functools.partial(
        decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id()
    )

    return model, tokenizer, params, decode


# Create a pytree mask of the trainable params.
def is_trainable_param(name, param):  # pylint: disable=unused-argument
    if name.startswith("llm/layers/attn/"):
        return True
    if name.startswith("llm/"):
        return False
    if name.startswith("img/"):
        return False
    raise ValueError(f"Unexpected param name {name}")


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


def preprocess_tokens(prefix, suffix=None, seqlen=None):
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


def postprocess_tokens(tokens):
    tokens = tokens.tolist()  # np.array to list[int]
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)  # type: ignore


@functools.partial(jax.jit, donate_argnums=(0,))
def update_fn(params, batch, learning_rate):
    imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

    def loss_fn(params):
        text_logits, _ = model.apply(
            {"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True
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
        example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)  # sum across seq_len.
        example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)  # weight by num of tokens.

        # batch_loss: mean of per example loss.
        return jnp.mean(example_loss)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Apply gradients to trainable params using SGD.
    def apply_grad(param, gradient, trainable):
        if not trainable:
            return param
        return param - learning_rate * gradient

    params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)
    return params, loss


@functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def maybe_cast_to_f32(params, trainable):
    return jax.tree.map(
        lambda p, m: p.astype(jnp.float32) if m else p, params, trainable
    )


def apply_params_sharding(params, params_sharding, trainable_mask):
    params, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    trainable_leaves = jax.tree.leaves(trainable_mask)
    for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
        params[idx] = big_vision.utils.reshard(params[idx], sharding)
        params[idx] = maybe_cast_to_f32(params[idx], trainable)
        params[idx].block_until_ready()
    params = jax.tree.unflatten(treedef, params)

    # Print params to show what the model is made of.
    def parameter_overview(params):
        for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
            print(f"{path:80s} {str(arr.shape):22s} {arr.dtype}")

    print(" == Model params == ")
    parameter_overview(params)
    return params


def train_data_iterator():
    """Never ending iterator over training examples."""
    # Shuffle examples and repeat so one can train for many epochs.
    dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
    for example in dataset.as_numpy_iterator():
        image = Image.open(io.BytesIO(example["image"]))  # type: ignore
        image = preprocess_image(image)

        # prefix = "caption en"  # Could also be a different prefix per example.
        prefix = example["prefix"].decode().lower()  # type: ignore
        suffix = example["suffix"].decode().lower()  # type: ignore
        tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix, SEQLEN)
        label, _, _, _ = preprocess_tokens(suffix, seqlen=SEQLEN)

        yield {
            "image": np.asarray(image),
            "text": np.asarray(tokens),
            "label": np.asarray(label),
            "mask_ar": np.asarray(mask_ar),
            "mask_loss": np.asarray(mask_loss),
        }


def validation_data_iterator():
    """Single iterator over validation examples."""
    for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
        image = Image.open(io.BytesIO(example["image"]))  # type: ignore
        image = preprocess_image(image)

        prefix = example["prefix"].decode().lower()  # type: ignore
        suffix = example["suffix"].decode().lower()  # type: ignore
        tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
        label, _, _, _ = preprocess_tokens(suffix, seqlen=SEQLEN)

        yield {
            "image": np.asarray(image),
            "text": np.asarray(tokens),
            "label": np.asarray(label),
            "mask_ar": np.asarray(mask_ar),
            "mask_input": np.asarray(mask_input),
        }


def render_inline(image, resize=(128, 128)):
    """Convert image into inline html."""
    image = Image.fromarray(image)
    image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format="jpeg")
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"


def render_example(image, caption):
    image = ((image + 1) / 2 * 255).astype(np.uint8)  # [-1,1] -> [0, 255]
    return f"""
    <div style="display: inline-flex; align-items: center; justify-content: center;">
        <img style="width:128px; height:128px;" src="{render_inline(image, resize=(64,64))}" />
        <p style="width:256px; margin:10px; font-size:small;">{html.escape(caption)}</p>
    </div>
    """


def show_train_samples():
    html_out = ""
    for idx, example in zip(range(8), train_data_iterator()):
        caption = postprocess_tokens(example["text"])  # detokenize model input.
        html_out += render_example(example["image"], caption)

    print("Training examples")
    display(HTML(html_out))


def make_predictions(
    params,
    data_iterator,
    *,
    num_examples=None,
    batch_size=4,
    seqlen=SEQLEN,
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
        labels = [postprocess_tokens(e["label"]) for e in examples]
        responses = [postprocess_tokens(t) for t in tokens]

        # Append to html output.
        for example, label, response in zip(examples, labels, responses):
            outputs.append(
                (postprocess_tokens(example["text"]), example["image"], label, response)
            )
            if num_examples is not None and len(outputs) >= num_examples:
                return outputs


def train(params):
    BATCH_SIZE = 16
    TRAIN_EXAMPLES = 16 * 50
    LEARNING_RATE = 0.03

    TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
    EVAL_STEPS = TRAIN_STEPS // 4

    print("Train Steps: ", TRAIN_STEPS)
    print("Eval Steps: ", EVAL_STEPS)

    train_data_it = train_data_iterator()

    sched_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=TRAIN_STEPS + 1,
        base=LEARNING_RATE,
        decay_type="cosine",
        warmup_percent=0.10,
    )

    for step in range(1, TRAIN_STEPS + 1):
        # Make list of N training examples.
        examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

        # Convert list of examples into a dict of np.arrays and load onto devices.
        batch = jax.tree.map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)

        # Training step and report training loss
        learning_rate = sched_fn(step)
        params, loss = update_fn(params, batch, learning_rate)

        loss = jax.device_get(loss)
        print(
            f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}"
        )

        if (step % EVAL_STEPS) == 0:
            print(f"Model predictions at step {step}")
            html_out = ""
            for prefix, image, _, caption in make_predictions(
                params, validation_data_iterator(), num_examples=4, batch_size=4
            ):
                html_out += render_example(image, prefix + caption)
            display(HTML(html_out))

    # Evaluation After training ...
    print("Model predictions")
    html_out = ""
    for prefix, image, _, caption in make_predictions(
        params, validation_data_iterator(), batch_size=4
    ):
        html_out += render_example(image, prefix + caption)
    display(HTML(html_out))


# %%
if __name__ == "__main__":
    init_devices()
    model, tokenizer, params, decode = load_pretrained_model()
    trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

    # Define Shardings & apply it to params
    mesh = jax.sharding.Mesh(jax.devices(), ("data"))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    params_sharding = big_vision.sharding.infer_sharding(
        params, strategy=[(".*", 'fsdp(axis="data")')], mesh=mesh
    )

    warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

    params = apply_params_sharding(
        params, params_sharding=params_sharding, trainable_mask=trainable_mask
    )

    # load datasets
    DATA_DIR_TRAIN = "./data/language_table_mini_train"
    train_dataset = big_vision.datasets.jsonl.DataSource(
        os.path.join(DATA_DIR_TRAIN, "_annotations.jsonl"),
        fopen_keys={"image": DATA_DIR_TRAIN},
    )

    DATA_DIR_VAL = "./data/language_table_mini_valid"
    val_dataset = big_vision.datasets.jsonl.DataSource(
        os.path.join(DATA_DIR_VAL, "_annotations.jsonl"),
        fopen_keys={"image": DATA_DIR_VAL},
    )

    show_train_samples()
    # %%
    train(params=params)

# %%
