import tensorflow as tf
from main import encoder_network, fully_connected_network


def load_mnist_data():
    return tf.keras.datasets.mnist.load_data()


def pre_process_data(img_data):
    img_data = img_data/255
    img_data = tf.keras.layers.Reshape(name='reshape_1', target_shape=[28, 28, 1])(img_data)
    return img_data


def ffc_model_without_autoencoder(inp_shape, num_classes):
    """

    :param inp_shape:
    :param num_classes:
    :return:
    """
    encoder_op, input_buffer = encoder_network(inp_shape=inp_shape)
    return fully_connected_network(num_classes=num_classes, encoded=encoder_op,
                                   inp_buff=input_buffer)


if __name__ == '__main__':
    print("starting.......")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train = pre_process_data(x_train)
    x_test = pre_process_data(x_test)

    print(y_train.shape)

    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    print('train data shape: ', x_train.shape, y_train.shape)
    print('test data shape: ', x_test.shape, y_test.shape)

    """
    hyper-params
    """
    batch_size = 32
    epochs = 10
    num_class = 10
    learning_rate = 0.003
    validation_split = 0.05

    ffc_model = ffc_model_without_autoencoder(inp_shape=[28, 28, 1], num_classes=num_class)

    ffc_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

    history = ffc_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                            validation_split=validation_split)

    print("train scores:")
    print(ffc_model.evaluate(x_train, y_train))
    print("test scores:")
    print(ffc_model.evaluate(x_test, y_test))

    ffc_model.save(filepath='./fc_wo_autoencoder', overwrite=True)
