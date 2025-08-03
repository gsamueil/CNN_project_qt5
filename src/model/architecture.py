import tensorflow as tf

def build_custom_cnn(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(200, kernel_size=(3,3), padding='same', activation='tanh', input_shape=input_shape),
        tf.keras.layers.Conv2D(180, kernel_size=(3,3), padding='same', activation='tanh'),
        tf.keras.layers.Conv2D(160, kernel_size=(3,3), padding='same', activation='tanh'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(140, kernel_size=(3,3), padding='same', activation='tanh'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
