import tensorflow as tf

import argparse
import datetime
import pprint

from sklearn.model_selection import train_test_split

from src.callbacks import get_callback
from src.dataset import build_dataset
from src.trainer import MyArcFace
from src.utils import load_data


IMAGE_SIZE = (280, 280, 3)


def define_argparser():
    p = argparse.ArgumentParser()

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
        type=int,
        default=1e-3,
        help=" ".join([
            "Learning rate.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--alpha", 
        type=int,
        default=30,
        help=" ".join([
            "Decay rate of learning rate.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--epochs", 
        type=int,
        default=1e-3,
        help=" ".join([
            "Epochs",
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

    ## Load dataset.
    inp, tar = load_data(config.data)

    ## Split the test dataset.
    tr_inp, _, tr_tar, _ = train_test_split(inp, tar, test_size=config.test_size, random_state=config.random_state)
    tr_inp, vl_inp, tr_tar, vl_tar = train_test_split(tr_inp, tr_tar, test_size=config.test_size, random_state=config.random_state)

    ## Make dataset.
    tr_ds = build_dataset(
        inp=tr_inp,
        tar=tr_tar,
        cache=True,
        shuffle=True,
        buffer_size=config.buffer_size, 
        batch_size=config.batch_size, 
        aug_type="imgaug",
    )
    vl_ds = build_dataset(
        inp=vl_inp,
        tar=vl_tar,
        cache=True,
        shuffle=False,
        buffer_size=None, 
        batch_size=config.batch_size, 
        aug_type=None,
    )

    ## Load model.
    model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    model = MyArcFace(
        input_shape=IMAGE_SIZE,
        name=model_name,
    )

    ## Compile.
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy(),
        optimizer=tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(
                learning_rate=config.lr,
            ),
        ),
    )

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
        verbose=1,
    )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
