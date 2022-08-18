import tensorflow as tf


class CNNBaseline(tf.keras.Model):
    def __init__(self, input_shape, n_labels):
        super().__init__(self)
        self.rescaling = tf.keras.layers.Rescaling(scale=1./255)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.2)

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.2)

        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_labels, activation='softmax')

    def call(self, inputs, training=True):
        x = inputs
        x = self.rescaling(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
