
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.python.eager import context


class ConditionedDense(Layer):
    '''
    For input x, and parameter p, this layer computes:
    y = sigma((a * p + b)) * x
    where a and b a trainable parameters, and sigma is an activation function
    '''
    def __init__(self,
                 channels,
                 name=None,
                 activation=None,
                 use_bias=False,
                 use_x_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout=0,
                 **kwargs):

        super().__init__(name=name, activity_regularizer=activity_regularizer, **kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.use_x_bias = use_x_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = dropout
        self.supports_masking = False

    def build(self, input_shape):
        c_in = input_shape[0][-1]
        c_out = self.channels
        self.kernel_network = Dense(c_in * c_out, use_bias=self.use_bias)
        if self.use_x_bias:
            self.x_bias = tf.Variable(tf.zeros([c_out], dtype=tf.float32), trainable=True)
        self.built = True
    
    def call(self, inputs):
        X = inputs[0]
        P = inputs[1]

        ndim = len(X.shape)
        X_shape = [tf.shape(X)[k] for k in range(ndim)]
        
        c_in = X.shape[-1]
        c_out = self.channels
            
        A = self.kernel_network(P)
        target_shape = X_shape[:-1]+[c_in, c_out]

        W = tf.reshape(A, target_shape)

        if self.dropout>0:
            W = Dropout(self.dropout)(W)

        if self.activation is not None:
            W = self.activation(W)
            
        if ndim == 2:
            output = tf.einsum('ij,ijk->ik', X, W)
        elif ndim == 3:
            output = tf.einsum('ijk,ijkl->ijl', X, W)
        else:
            raise Exception('Only ndim = 2 and 3 inputs are supported')

        if self.use_x_bias:
            output = tf.nn.bias_add(output, self.x_bias)

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
