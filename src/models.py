import tensorflow as tf


def bn_relu_conv2d(x: tf.Tensor, filters: int, kernel_size: int) -> tf.Tensor:
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
    
    return x


def transition_block(x: tf.Tensor) -> tf.Tensor:
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(x.shape[-1] // 2, 1, padding = "same")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), strides=2)(x)

    return x


def dense_block(x: tf.Tensor, num_conv: int, growth_rate: int):
    ## Densely connect.
    for _ in range(num_conv):
        residual = x
        x = bn_relu_conv2d(x, 4 * growth_rate, 1)
        x = bn_relu_conv2d(x, growth_rate, 3)
        x = tf.keras.layers.Concatenate(axis=-1)([x, residual])

    return x


def embedding_model(input_shape: tuple, embedding_dim: int = 512, model_name: str = "embedding_model", growth_rate: int = 32) -> tf.keras.Model:
    x = model_input = tf.keras.layers.Input(input_shape)

    ## Entry Flow
    x = tf.keras.layers.Conv2D(2 * growth_rate, 7, strides=2, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding = "same")(x)

    ## Middle Flow
    for i, num_conv in enumerate([6, 12, 24, 16]):
        x = dense_block(x, num_conv, growth_rate)
        if i != 3: 
            x = transition_block(x)

    ## Exit Flow
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)

    ## L2-normalize.
    model_output = x = tf.keras.layers.Lambda(lambda xx: tf.math.l2_normalize(xx, axis=-1))(x)

    return tf.keras.Model(
        inputs=model_input,
        outputs=model_output,
        name=model_name,
    )
