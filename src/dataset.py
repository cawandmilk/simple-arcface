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
) -> tf.data.Dataset:
    ## Assertion.
    assert aug_type in [None, "imgaug"]
    rand_aug = iaa.RandAugment(n=3, m=7)

    @tf.function
    def normalize(inp_, tar_):
        inp_ = tf.cast(inp_, dtype=tf.float32) / 255.
        return inp_, tar_

    def aug_fn(inp_, tar_):
        ## Ref: https://keras.io/examples/vision/randaugment/
        def aug_py_fn(inp_, tar_):
            inp_ = tf.cast(inp_, tf.uint8)
            return rand_aug(inp_=inp_.numpy()), tar_

        return tf.py_function(
            aug_py_fn, 
            inp=[inp_, tar_],
            Tout=[tf.uint8, tf.int32]
        )

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
