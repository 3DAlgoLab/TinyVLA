# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

# %%

tf.config.set_visible_devices([], "GPU")


data_dir = "/tmp/tfds"


print(tfds.list_builders())


data, info = tfds.load(
    name="cats_vs_dogs",
    data_dir=data_dir,
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
)

(cats_dogs_data_train, cast_dogs_data_test) = data

WIDTH, HEIGHT = 200, 200
NUM_LABELS = info.features["label"].num_classes


def preprocess(img, label):
    return tf.image.resize(img, [HEIGHT, WIDTH]) / 255.0, label


train_data = tfds.as_numpy(cats_dogs_data_train.map(preprocess).batch(32).prefetch(1))
test_data = tfds.as_numpy(cast_dogs_data_test.map(preprocess).batch(32).prefetch(1))

# %%

plt.rcParams["figure.figsize"] = [20, 10]

CLASS_NAMES = ["cat", "dog"]
ROWS = 3
COLS = 8

i = 0
fig, ax = plt.subplots(ROWS, COLS)
processed = cats_dogs_data_train.map(preprocess)

for image, label in processed.take(ROWS * COLS):
    ax[int(i / COLS), i % COLS].axis("off")
    ax[int(i / COLS), i % COLS].set_title(CLASS_NAMES[label])
    ax[int(i / COLS), i % COLS].imshow(image)
    i += 1

plt.show()

# %%
import jax

batch_images, batch_labels = next(iter(train_data))
batch_images[0].shape
# %%
image = batch_images[0]
image.min(), image.max()
# %%
seed = 42
key = jax.random.PRNGKey(seed)

std_noise = jax.random.normal(key, image.shape)
std_noise.min(), std_noise.max()
# %%
noise = 0.5 + 0.1 * std_noise
plt.imshow(noise)
# %%
new_image = image + noise
new_image.min(), new_image.max()
# %%
plt.imshow(new_image)
# %%
new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min())
plt.imshow(new_image)
# %%
new_image.min(), new_image.max()
# %%
from jax import random

key = random.PRNGKey(42)
key
# %%
key1, key2 = random.split(key, num=2)
# %%
