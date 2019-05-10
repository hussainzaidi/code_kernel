import tensorflow as tf

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        #self.w = self.add_weight(shape=(input_shape[-1], self.units),initializer='random_normal',trainable=True, name='weights')
        #self.b = self.add_weight(shape=(self.units,),initializer='random_normal',trainable=True, name='biases')
        #self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.units])
        self.kernel = self.add_variable("kernel", shape=[1, self.units], initializer='one')

    def call(self, inputs):
        #return tf.matmul(inputs, self.kernel)
        #return tf.mul(inputs, self.kernel)
        return self.kernel*inputs

