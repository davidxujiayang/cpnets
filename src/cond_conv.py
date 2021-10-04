import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Lambda, Conv2D
from tensorflow.python.eager import context


class ConditionedConvolution2D(Layer):
    def __init__(self,
                 channels,
                 kernel_shape=(3, 3),
                 padding='SAME',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.channels = channels
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        kernel_shape = self.kernel_shape
        c_in = input_shape[0][-1]
        c_out = self.channels
        self.kernel_network = Dense(kernel_shape[0] * kernel_shape[1] * c_in * c_out, use_bias=self.use_bias, name='weight_out', activation=self.activation)

        self.built = True
    
    def call(self, inputs):
        X = inputs[0]
        P = inputs[1]

        kernel_shape = self.kernel_shape

        ndim = len(X.shape)
        X_shape = [tf.shape(X)[k] for k in range(ndim)]
        
        c_in = X.shape[-1]
        c_out = self.channels

        A = self.kernel_network(P)
        target_shape = X_shape[0:1]+[kernel_shape[0], kernel_shape[1], c_in, c_out]
        W = tf.reshape(A, target_shape)

        def myfunc(args):
            x = tf.expand_dims(args[0], axis=0)
            w = args[1]
            res = tf.nn.conv2d(x, w, padding=self.padding, strides=1)[0,]
            return res

        output = tf.map_fn(myfunc, (X, W), dtype=X.dtype)
        # print(output.get_shape().as_list())


        if not context.executing_eagerly():
            shape = X.shape.as_list()
            output_shape = shape[:-1] + [c_out]
            output.set_shape(output_shape)

        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))