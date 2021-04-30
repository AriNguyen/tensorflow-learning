import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers


def CNN_model():
    """ 6 convolutional layer + 1 Pooling layer

    :param input_x:
    :return:
    """
    model = tf.keras.Sequential(
        [
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            layers.Activation('relu'),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2)),
            layers.Activation('relu'),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            layers.Activation('relu'),

            layers.GlobalAveragePooling2D(),
            layers.Dense(32),
            layers.Activation('relu'),
            layers.Dense(10),
            layers.Activation('softmax')
        ]
    )

    return model


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    IMG_SIZE = (28, 28, 1)
    input_img = layers.Input(shape=IMG_SIZE)

    # Build model
    CNN_model = CNN_model()
    output = CNN_model(input_img)
    print(CNN_model.summary())

    #
