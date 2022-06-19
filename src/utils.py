import collections
import cv2

import numpy as np

from pathlib import Path

from prepare import get_target_items_and_urls


def load_data(data_dir):
    ## Load item names.
    item_names = get_target_items_and_urls().keys()

    ## Glob.
    print(f"Deduplicating Results:")
    return_values = collections.OrderedDict({})

    for item_name in item_names:
        ## Load.
        fpaths = sorted(list(Path(data_dir, item_name).glob("*.jpeg")))
        images = np.unique([cv2.imread(i) for i in fpaths], axis=0)
        return_values[item_name] = images

        ## Print.
        ratio = (images.shape[0] - len(fpaths)) / images.shape[0] * 100
        print(f"  - {item_name}: {len(fpaths)} -> {images.shape[0]} ({ratio:.1f})")

    x = np.stack(return_values.values(), axis=0)
    y = np.stack([np.full_like(i, i.shape[0]) for i in return_values.values()], axis=0)

    return x, y
