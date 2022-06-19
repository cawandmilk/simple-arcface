import tensorflow as tf

from src.models import embedding_model


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
        x = tf.math.cos(tf.math.acos(x) + self.m)
        x = self.scaling_fn(x)

        return x


    @tf.function
    def train_step(self, x: tf.Tensor) -> dict:
        ## Unpack.
        inp, tar = x

        with tf.GradientTape() as tape:
            ## Forward.
            y_pred = self(inp, training=True)

            ## Calculate loss.
            loss_value = self.compiled_loss(y_true=tar, y_pred=y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss_value)

        ## Backword.
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ## Update metrics.
        self.compiled_metrics.update_state(y_true=tar, y_pred=y_pred)

        return {m.name: m.result() for m in self.metrics}


    @tf.function
    def test_step(self, x: tf.Tensor):
        ## Unpack.
        inp, tar = x

        ## Forward.
        y_pred = self(inp, training=True)

        ## Calculate loss.
        loss_value = self.compiled_loss(y_true=tar, y_pred=y_pred)
        scaled_loss = self.optimizer.get_scaled_loss(loss_value)

        ## Update metrics.
        self.compiled_metrics.update_state(y_true=tar, y_pred=y_pred)

        return {m.name: m.result() for m in self.metrics}
