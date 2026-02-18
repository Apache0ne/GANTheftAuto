import tensorflow as tf


def _l2_normalize(x, axis=None, epsilon=1e-12):
    square_sum = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
    return x * tf.math.rsqrt(tf.maximum(square_sum, epsilon))


def _deserialize_layer(layer_config):
    if isinstance(layer_config, tf.keras.layers.Layer):
        return layer_config

    if not isinstance(layer_config, dict):
        raise TypeError(f"Unsupported wrapped layer config type: {type(layer_config)}")

    class_name = layer_config.get("class_name")
    config = layer_config.get("config", {})

    # Keras 3 can fail to locate legacy class names in this context.
    layer_cls = getattr(tf.keras.layers, class_name, None)
    if layer_cls is not None and hasattr(layer_cls, "from_config"):
        return layer_cls.from_config(config)

    return tf.keras.layers.deserialize(layer_config, custom_objects=get_custom_objects())


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        groups=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.groups = groups
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        rank = len(input_shape)
        axis = self.axis if self.axis >= 0 else rank + self.axis
        dim = input_shape[axis]
        if dim is None:
            raise ValueError("Channel dimension must be defined for InstanceNormalization.")

        weight_shape = (int(dim),)
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=weight_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
            )
        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=weight_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        rank = len(inputs.shape)
        if rank is None:
            raise ValueError("Input rank must be known for InstanceNormalization.")

        axis = self.axis if self.axis >= 0 else rank + self.axis
        reduction_axes = [i for i in range(1, rank) if i != axis]
        mean, variance = tf.nn.moments(inputs, axes=reduction_axes, keepdims=True)
        outputs = (inputs - mean) * tf.math.rsqrt(variance + self.epsilon)

        broadcast_shape = [1] * rank
        broadcast_shape[axis] = tf.shape(inputs)[axis]
        if self.scale and self.gamma is not None:
            outputs = outputs * tf.reshape(self.gamma, broadcast_shape)
        if self.center and self.beta is not None:
            outputs = outputs + tf.reshape(self.beta, broadcast_shape)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "groups": self.groups,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
                "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
                "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
                "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
                "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
                "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            }
        )
        return config


class TFOpLambda(tf.keras.layers.Layer):
    def __init__(self, function=None, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 0:
                raise ValueError("TFOpLambda received an empty input list.")
            output = inputs[0]
            for item in inputs[1:]:
                output = tf.add(output, item)
            return output
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"function": self.function})
        return config


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = int(power_iterations)
        self._u = None

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        if not hasattr(self.layer, "kernel"):
            raise ValueError("SpectralNormalization expects wrapped layer to have a `kernel` attribute.")

        self.kernel = self.layer.kernel
        kernel_shape = self.kernel.shape.as_list()
        self._kernel_shape = kernel_shape
        self._u = self.add_weight(
            shape=(1, kernel_shape[-1]),
            initializer=tf.keras.initializers.RandomNormal(0, 1),
            trainable=False,
            name="sn_u",
            dtype=self.kernel.dtype,
        )
        super().build(input_shape)

    def _normalized_kernel_and_u(self):
        w = tf.reshape(self.kernel, [-1, self._kernel_shape[-1]])
        u = self._u
        for _ in range(self.power_iterations):
            v = _l2_normalize(tf.matmul(u, w, transpose_b=True))
            u = _l2_normalize(tf.matmul(v, w))
        sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
        w_norm = tf.reshape(w / sigma, self._kernel_shape)
        return w_norm, u

    def _conv2d_with_kernel(self, inputs, kernel):
        if self.layer.data_format == "channels_first":
            data_format = "NCHW"
            strides = [1, 1, self.layer.strides[0], self.layer.strides[1]]
            dilations = [1, 1, self.layer.dilation_rate[0], self.layer.dilation_rate[1]]
        else:
            data_format = "NHWC"
            strides = [1, self.layer.strides[0], self.layer.strides[1], 1]
            dilations = [1, self.layer.dilation_rate[0], self.layer.dilation_rate[1], 1]

        outputs = tf.nn.conv2d(
            input=inputs,
            filters=kernel,
            strides=strides,
            padding=self.layer.padding.upper(),
            data_format=data_format,
            dilations=dilations,
        )
        if self.layer.use_bias:
            outputs = tf.nn.bias_add(outputs, self.layer.bias, data_format=data_format)
        if self.layer.activation is not None:
            outputs = self.layer.activation(outputs)
        return outputs

    def call(self, inputs, training=None):
        kernel, u = self._normalized_kernel_and_u()
        if training and tf.executing_eagerly():
            self._u.assign(u)

        if isinstance(self.layer, tf.keras.layers.Conv2D):
            return self._conv2d_with_kernel(inputs, kernel)
        return self.layer(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power_iterations": self.power_iterations,
                "layer": tf.keras.layers.serialize(self.layer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        layer_config = config.pop("layer")
        layer = _deserialize_layer(layer_config)
        return cls(layer=layer, **config)


def get_custom_objects():
    return {
        "InstanceNormalization": InstanceNormalization,
        "Addons>InstanceNormalization": InstanceNormalization,
        "SpectralNormalization": SpectralNormalization,
        "Addons>SpectralNormalization": SpectralNormalization,
        "TFOpLambda": TFOpLambda,
    }
