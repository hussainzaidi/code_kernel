import tensorflow as tf

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[1, self.units], initializer='one')

    def call(self, inputs):
        return self.kernel*inputs

