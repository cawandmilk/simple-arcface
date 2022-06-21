import tensorflow as tf
import tensorflow_addons as tfa

import argparse
import datetime
import os
import pprint

import numpy as np

from sklearn.model_selection import train_test_split

from src.callbacks import get_callback
from src.dataset import build_dataset
from src.models import MyBaseline, MyArcFace
from src.utils import load_data, set_gpu_memory_growthable, set_mixed_precision_policy

## Hide the logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IMAGE_SIZE = (280, 280, 3)


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_name", 
        type=str,
        required=True,
        help=" ".join([
            "The model name."
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--is_baseline", 
        action="store_true",
        help=" ".join([
            "Whether using arcface or not."
            "Default=%(default)s",
        ]),
    )

    p.add_argument(
        "--data", 
        type=str,
        default="data",
        help=" ".join([
            "The path you saved the images."
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--test_size", 
        type=float,
        default=0.2,
        help=" ".join([
            "Test (or validation) size.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--random_state", 
        type=int,
        default=42,
        help=" ".join([
            "The seed",
            "Default=%(default)s",
        ]),
    )

    p.add_argument(
        "--buffer_size", 
        type=int,
        default=30_000,
        help=" ".join([
            "Buffer size to shuffle your dataset.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--batch_size", 
        type=int,
        default=32,
        help=" ".join([
            "Batch size.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--lr", 
        type=float,
        default=3e-4,
        help=" ".join([
            "Learning rate.",
            "Default=%(default)s",
        ]),
    )
    # p.add_argument(
    #     "--alpha", 
    #     type=int,
    #     default=1./20,
    #     help=" ".join([
    #         "Decay rate of learning rate.",
    #         "Default=%(default)s",
    #     ]),
    # )
    p.add_argument(
        "--epochs", 
        type=int,
        default=200,
        help=" ".join([
            "Epochs",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--augment", 
        action="store_true",
        help=" ".join([
            "Whether applying augmentation or not.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--logs", 
        type=str,
        default="logs",
        help=" ".join([
            "The directory that the logs will save.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--ckpt", 
        type=str,
        default="ckpt",
        help=" ".join([
            "The directory that the checkpoints will save.",
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()
    return config


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Set gpu memory growthable.
    set_gpu_memory_growthable()

    ## Apply mixed precision policy.
    set_mixed_precision_policy()

    ## Load dataset.
    inp, tar = load_data(config.data, image_size=IMAGE_SIZE)

    ## Split the test dataset.
    tr_inp, ts_inp, tr_tar, ts_tar = train_test_split(
        inp, tar, 
        test_size=config.test_size, 
        random_state=config.random_state,
        stratify=tar,
    )
    tr_inp, vl_inp, tr_tar, vl_tar = train_test_split(
        tr_inp, tr_tar, 
        test_size=config.test_size, 
        random_state=config.random_state,
        stratify=tr_tar,
    )
    print(f"|train|={tr_inp.shape[0]}, |valid|={vl_inp.shape[0]}, |test|={ts_inp.shape[0]}")

    ## Make dataset.
    tr_ds = build_dataset(
        inp=tr_inp,
        tar=tr_tar,
        cache=True,
        shuffle=True,
        buffer_size=config.buffer_size, 
        batch_size=config.batch_size, 
        aug_type="imgaug" if config.augment else None,
        image_size=IMAGE_SIZE,
    )
    vl_ds = build_dataset(
        inp=vl_inp,
        tar=vl_tar,
        cache=True,
        shuffle=False,
        buffer_size=None, 
        batch_size=config.batch_size, 
        aug_type=None,
        image_size=IMAGE_SIZE,
    )
    ts_ds = build_dataset(
        inp=ts_inp,
        tar=ts_tar,
        cache=False,
        shuffle=False,
        buffer_size=None, 
        batch_size=config.batch_size, 
        aug_type=None,
        image_size=IMAGE_SIZE,
    )
    print(f"tr_ds: {repr(tr_ds)}")
    print(f"vl_ds: {repr(vl_ds)}")
    print(f"ts_ds: {repr(ts_ds)}")

    ## Load model.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "-".join([config.model_name, nowtime])

    if config.is_baseline:
        ## Only softmax.
        model = MyBaseline(
            input_shape=IMAGE_SIZE,
            name=model_name,
        )
    else:
        model = MyArcFace(
            input_shape=IMAGE_SIZE,
            name=model_name,
        )

    ## Compile.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["acc"],
    )
    model.build([None, *IMAGE_SIZE])
    model.summary()

    ## Just train =)
    model.fit(
        tr_ds,
        validation_data=vl_ds,
        epochs=config.epochs,
        callbacks=get_callback(
            file_name=model_name,
            init_lr=config.lr,
            epochs=config.epochs,
            alpha=config.alpha,
            logs=config.logs,
            ckpt=config.ckpt,
        ),
        verbose=0,
    )

    ## Evaluate.
    model.evaluate(ts_ds, verbose=1)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
