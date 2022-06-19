import tensorflow as tf

import os

from typing import List


def get_callback(
    file_name: str, 
    init_lr: float,
    epochs: int,
    alpha: int = 30,
    logs: str = "logs", 
    ckpt: str = "ckpt",
) -> List:
    ## Gatter iterm.
    callbacks = []

    ## Tensorboard logging.
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logs, file_name),
            update_freq="batch",
        )
    )

    ## Checkpoint.
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            log_dir=os.path.join(ckpt, file_name),
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            mode="auto",
            save_freq="epoch",
        )
    )

    ## Learning rate scheudler.
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(
            schedule=tf.keras.optimizers.schedules.CosineDecay (
                initial_learning_rate=init_lr,
                decay_steps=epochs,
                alpha=alpha,
            ),
            verbose=0,
        )
    )

    return callbacks