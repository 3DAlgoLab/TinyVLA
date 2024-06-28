import os 
os.environ['XLA_FLAGS']='--xla_force_host_platform_device_count=8'

import jax.numpy as jnp
from jax import lax 
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp

def predict(w1, w2, images):
    hiddens = relu(jnp.dot(images, w1))
    logits = jnp.dot(hiddens, w2)
    return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(w1, w2, images, labels):
    predictions = predict(w1, w2, images)
    targets = one_hot(labels, predictions.shape[-1])
    losses = jnp.sum(targets * predictions, axis=1)
    return -jnp.mean(losses, axis=0)

w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784))
labels = jnp.zeros(128, dtype=jnp.int32)

print(loss(w1, w2, images, labels))