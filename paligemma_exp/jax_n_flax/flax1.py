# %%
from multiprocessing.pool import INIT
import jax
from jax import random
from flax import linen as nn
from icecream import ic
from jax import numpy as jnp

INIT_LR = 0.01
DECAY_RATE = 0.95
DECAY_STEPS = 5
# %%
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], device_type="GPU")

data_dir = "/tmp/tfds"
data, info = tfds.load("mnist", with_info=True, data_dir=data_dir, as_supervised=True)
data_train = data["train"]
data_test = data["test"]


# %%
info


# %%
def preprocess(img, label):
    return (tf.cast(img, tf.float32) / 255.0), label


data_train_vis = data_train.map(preprocess)

# %%
import matplotlib.pyplot as plt
import numpy as np

ROWS = 3
COLS = 10
HEIGHT = 28
WIDTH = 28
CHANNELS = 1
NUM_PIXELS = HEIGHT * WIDTH * CHANNELS
NUM_LABELS = info.features["label"].num_classes


i = 0
fig, ax = plt.subplots(ROWS, COLS)
for image, label in data_train_vis.take(ROWS * COLS):
    ax[int(i / COLS), i % COLS].axis("off")
    ax[int(i / COLS), i % COLS].set_title(str(label.numpy()))
    ax[int(i / COLS), i % COLS].imshow(np.reshape(image, (28, 28)), cmap="gray")
    i += 1

plt.show()


# %%
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=512)(x)
        # x = nn.activation.swish(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


model = MLP()

key1, key2 = random.split(random.PRNGKey(0))
random_flattened_image = random.normal(key1, (28 * 28 * 1,))
params = model.init(key2, random_flattened_image)

info = jax.tree_util.tree_map(lambda x: x.shape, params)
ic(info)


result = model.apply(params, random_flattened_image)
ic(result)

# %%
print(model.tabulate(key2, random_flattened_image))

# %%


def loss(params, images, targets):
    """Categorical crossentropy loss."""
    logits = model.apply(params, images)
    log_preds = logits - jax.nn.logsumexp(logits)  # type: ignore
    return -jnp.mean(jnp.sum(log_preds * targets))


@jax.jit
def update(params, x, y, epoch_number):
    loss_value, grads = jax.value_and_grad(loss)(params, x, y)
    lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)

    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads), loss_value


@jax.jit
def batch_accuracy(params, images, targets):
    images = jnp.reshape(images, (len(images), NUM_PIXELS))
    predicted_class = jnp.argmax(model.apply(params, images), axis=1)  # new
    return jnp.mean(predicted_class == targets)


def accuracy(params, data):
    accs = []
    for images, targets in data:
        accs.append(batch_accuracy(params, images, targets))
    return jnp.mean(jnp.array(accs))


# %%
import time

train_data = tfds.as_numpy(data_train.map(preprocess).batch(32).prefetch(1))
test_data = tfds.as_numpy(data_test.map(preprocess).batch(32).prefetch(1))

num_epochs = 25

for epoch in range(num_epochs):
    start_time = time.time()
    losses = []
    for x, y in train_data:
        x = jnp.reshape(x, (len(x), NUM_PIXELS))
        y = jax.nn.one_hot(y, NUM_LABELS)
        params, loss_value = update(params, x, y, epoch)
        losses.append(loss_value)
    epoch_time = time.time() - start_time

    start_time = time.time()
    train_acc = accuracy(params, train_data)
    test_acc = accuracy(params, test_data)
    eval_time = time.time() - start_time
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Eval in {:0.2f} sec".format(eval_time))
    print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

# %%
