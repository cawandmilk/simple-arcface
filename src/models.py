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


def embedding_model(
    input_shape: tuple, 
    embedding_dim: int = 512, 
    model_name: str = "embedding_model", 
    growth_rate: int = 24,
    normalize: bool = True,
) -> tf.keras.Model:
    ## Define a input.
    x = model_input = tf.keras.layers.Input(input_shape)

    ## Entry Flow
    x = tf.keras.layers.Conv2D(2 * growth_rate, 7, strides=2, padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    ## Middle Flow
    num_layers = [6, 12, 8] # [6, 12, 24, 16]
    for i, num_conv in enumerate(num_layers):
        x = dense_block(x, num_conv, growth_rate)
        if i != len(num_layers) - 1: 
            x = transition_block(x)

    ## Exit Flow
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)

    if normalize:
        ## L2-normalize.
        model_output = x = tf.keras.layers.Lambda(lambda xx: tf.math.l2_normalize(xx, axis=-1))(x)
    else:
        model_output = x = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(
        inputs=model_input,
        outputs=model_output,
        name=model_name,
    )


class MyBaseline(tf.keras.Model):

    def __init__(
        self,
        input_shape: tuple = (280, 280, 3),
        embedding_dim: int = 512,
        scaling_factor: int = 64,
        num_classes: int = 9,
        name: str = None,
        **kwargs,
    ):
        super(MyBaseline, self).__init__(name=name, **kwargs)
        self.embedding_model = embedding_model(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            normalize=False,
        )
        self.fc = tf.keras.layers.Dense(num_classes)


    @tf.function
    def call(self, inp: tf.Tensor, training: bool = None) -> tf.Tensor:
        ## Embed features.
        x = self.embedding_model(inp, training=training)
        ## |x| = (batch_size, embedding_dim)

        ## Calcualte cosine similarity.
        x = self.fc(x)
        ## |x| = (batch_size, num_classes)

        return x


class MyArcFace(tf.keras.Model):

    def __init__(
        self,
        input_shape: tuple = (280, 280, 3),
        embedding_dim: int = 512,
        scaling_factor: int = 64,
        num_classes: int = 9,
        m: int = 20,
        w: int = 10,
        b: int = -5,
        name: str = None,
        **kwargs,
    ):
        super(MyArcFace, self).__init__(name=name, **kwargs)
        self.embedding_model = embedding_model(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            normalize=True,
        )
        self.fc = tf.keras.layers.Dense(num_classes, use_bias=False)
        self.m = tf.cast(m, dtype=tf.float32)

        ## Weights for scaling cosine similarity.
        self.scaling_fn = tf.keras.layers.Lambda(lambda x: w * x + b)


    @tf.function
    def call(self, inp: tf.Tensor, training: bool = None) -> tf.Tensor:
        ## Embed features.
        x = self.embedding_model(inp, training=training)
        ## |x| = (batch_size, embedding_dim)

        ## Calcualte cosine similarity.
        x = self.fc(x)
        ## |x| = (batch_size, num_classes)

        ## Apply margin and sacle again.
        x = tf.clip_by_value(x, -1, 1) ## remove nan
        x = tf.math.cos(tf.math.acos(x) + tf.cast(self.m, dtype=x.dtype))
        x = self.scaling_fn(x)

        return x


    # @tf.function
    # def train_step(self, x: tf.Tensor) -> dict:
    #     ## Unpack.
    #     inp, tar = x

    #     with tf.GradientTape() as tape:
    #         ## Forward.
    #         y_pred = self(inp, training=True)

    #         ## Calculate loss.
    #         loss_value = self.compiled_loss(y_true=tar, y_pred=y_pred)
    #         scaled_loss = self.optimizer.get_scaled_loss(loss_value)

    #     ## Backword.
    #     scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
    #     gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    #     ## Update metrics.
    #     self.compiled_metrics.update_state(y_true=tar, y_pred=y_pred)

    #     return {m.name: m.result() for m in self.metrics}


    # @tf.function
    # def test_step(self, x: tf.Tensor):
    #     ## Unpack.
    #     inp, tar = x

    #     ## Forward.
    #     y_pred = self(inp, training=True)

    #     ## Calculate loss.
    #     loss_value = self.compiled_loss(y_true=tar, y_pred=y_pred)
    #     scaled_loss = self.optimizer.get_scaled_loss(loss_value)

    #     ## Update metrics.
    #     self.compiled_metrics.update_state(y_true=tar, y_pred=y_pred)

    #     return {m.name: m.result() for m in self.metrics}
