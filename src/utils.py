import tensorflow as tf

import collections
import cv2

import numpy as np

from pathlib import Path

from prepare import get_target_items_and_urls


def set_gpu_memory_growthable() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            ## Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
        except RuntimeError as e:
            ## Memory growth must be set before GPUs have been initialized
            print(e)


def set_mixed_precision_policy() -> None:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def load_data(data_dir: str, image_size: tuple = (280, 280, 3)):
    ## Load item names.
    item_names = get_target_items_and_urls().keys()

    ## Glob.
    print(f"Deduplicating Results:")
    return_values = collections.OrderedDict({})

    for item_name in item_names:
        ## Load.
        images = []

        fpaths = sorted(list(Path(data_dir, item_name).glob("*.jpeg")))
        for fpath in fpaths:
            image = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            ## Synthetics check.
            if image is not None and image.shape == image_size:
                ## Append.
                images.append(image)
        
        ## De-duplicating.
        images = np.stack(images, axis=0)
        images = np.unique(images, axis=0)

        return_values[item_name] = images

        ## Print.
        ratio = (images.shape[0] - len(fpaths)) / len(fpaths) * 100
        print(f"  - {item_name}: {len(fpaths)} -> {images.shape[0]} ({ratio:.1f}%)")

    ## Concatenate all.
    x = np.concatenate(list(return_values.values()), axis=0)
    y = np.concatenate([[n] * i.shape[0] for n, i in enumerate(return_values.values())]).astype(np.int32)

    return x, y
