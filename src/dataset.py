import tensorflow as tf

from imgaug import augmenters as iaa

import numpy as np


AUTO = tf.data.AUTOTUNE


def build_dataset(
    inp: np.ndarray,
    tar: np.ndarray,
    cache: bool = True,
    shuffle: bool = True,
    buffer_size: int = None, 
    batch_size: int = 32, 
    aug_type: str = None,
    image_size: tuple = (280, 280, 3),
) -> tf.data.Dataset:
    ## Assertion.
    assert aug_type in [None, "imgaug"]
    rand_aug = iaa.RandAugment(n=3, m=7)


    def aug_py_fn(inp_, tar_):
        ## Ref: https://keras.io/examples/vision/randaugment/
        inp_ = tf.cast(inp_, tf.uint8)
        return rand_aug(images=inp_.numpy()), tar_


    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, *image_size), dtype=tf.uint8),
    #     tf.TensorSpec(shape=(None,), dtype=tf.int32),
    # ])
    def aug_fn(inp_, tar_):
        return tf.py_function(
            aug_py_fn, 
            inp=[inp_, tar_],
            Tout=[tf.uint8, tf.int32]
        )


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, *image_size), dtype=tf.uint8),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ])
    def normalize(inp_, tar_):
        inp_ = tf.cast(inp_, dtype=tf.float32) / 255.
        return inp_, tar_


    ## Define augmentation components.
    ds = tf.data.Dataset.from_tensor_slices((inp, tar))

    ## Caching.
    if cache:
        ds = ds.cache()

    ## Shuffling.
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)

    ## Batch.
    ds = ds.batch(batch_size, num_parallel_calls=AUTO)

    ## Augmentation after batching.
    if aug_type == "imgaug":
        ds = ds.map(aug_fn, num_parallel_calls=AUTO)

    ## Normalize.
    ds = ds.map(normalize, num_parallel_calls=AUTO)

    ## Prefetch.
    ds = ds.prefetch(AUTO)

    ## Return.
    return ds
