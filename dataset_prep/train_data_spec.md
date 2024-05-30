
# Train Data Specification

```python
spec = {
    "episode_id": TensorSpec(shape=(), dtype=tf.string, name=None),
    "steps": DatasetSpec(
        {
            "action": TensorSpec(shape=(2,), dtype=tf.float32, name=None),
            "is_first": TensorSpec(shape=(), dtype=tf.bool, name=None),
            "is_last": TensorSpec(shape=(), dtype=tf.bool, name=None),
            "is_terminal": TensorSpec(shape=(), dtype=tf.bool, name=None),
            "observation": {
                "effector_target_translation": TensorSpec(
                    shape=(2,), dtype=tf.float32, name=None
                ),
                "effector_translation": TensorSpec(
                    shape=(2,), dtype=tf.float32, name=None
                ),
                "instruction": TensorSpec(shape=(512,), dtype=tf.int32, name=None),
                "rgb": TensorSpec(shape=(360, 640, 3), dtype=tf.uint8, name=None),
            },
            "reward": TensorSpec(shape=(), dtype=tf.float32, name=None),
        },
        TensorShape([]),
    ),
}
```