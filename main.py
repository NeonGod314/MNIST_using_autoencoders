import tensorflow as tf


# @tf.function
def load_mnist_data():
    return tf.keras.datasets.mnist.load_data()


# @tf.function
def pre_process_data(img_data):
    img_data = img_data/255
    img_data = tf.keras.layers.Reshape(name='reshape_1', target_shape=[28, 28, 1])(img_data)
    return img_data


def encoder_network(inp_shape):
    """
    Implements encdoer
    :return: op of endcoder
    """
    input_buffer = tf.keras.layers.Input(name='inp_buff', shape=inp_shape)

    conv1 = tf.keras.layers.Conv2D(name='conv1', filters=3, kernel_size=[3, 3], padding='same',
                                   activation='relu')(input_buffer)  # 28*28*3
    mxp1 = tf.keras.layers.MaxPool2D(name='mxp1')(conv1)  # 14*14*3
    conv2 = tf.keras.layers.Conv2D(name='conv2', filters=9, kernel_size=[3, 3], padding='same',
                                   activation='relu')(mxp1)  # 14*14*3
    mxp2 = tf.keras.layers.MaxPool2D(name='mxp2')(conv2)  # 7*7*3
    batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mxp2)  # 7*7*3
    conv3 = tf.keras.layers.Conv2D(name='conv3', filters=27, kernel_size=[3, 3], padding='same',
                                   activation='relu')(batch_norm1)  # 7*7*3
    batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm2')(conv3)  # 7*7*3

    return batch_norm2, input_buffer


def decoder_network(encoded):
    """
    Implements decoder network
    :return: decoded op
    """
    # encoded, inp_buff = encoder_network()
    conv4 = tf.keras.layers.Conv2D(name='conv4', filters=9, kernel_size=[3, 3], padding='same',
                                   activation='relu')(encoded)  # 7*7*3
    batch_norm3 = tf.keras.layers.BatchNormalization(name='batch_norm3')(conv4)  # 7*7*3
    up_1 = tf.keras.layers.UpSampling2D(name='up1', size=(2, 2))(batch_norm3)  # 14*14*3
    conv5 = tf.keras.layers.Conv2D(name='conv5', filters=3, kernel_size=[3, 3], padding='same',
                                   activation='relu')(up_1)  # 14*14*3
    batch_norm4 = tf.keras.layers.BatchNormalization(name='batch_norm4')(conv5)  # 14*14*3
    print(batch_norm4.shape)
    up_2 = tf.keras.layers.UpSampling2D(name='up2', size=[2, 2])(batch_norm4)  # 28*28*3
    decoded = tf.keras.layers.Conv2D(name='decoded', filters=1, kernel_size=[3, 3], activation='linear', padding='same')(up_2)

    return decoded


# @tf.function
def auto_encoder_model(inp_shape):
    encoded, inp_buff = encoder_network(inp_shape=inp_shape)
    decoded = decoder_network(encoded=encoded)
    return tf.keras.Model(inputs=inp_buff, outputs=decoded, name='autoencoder')


# @tf.function
def fully_connected_network(num_classes, encoded, inp_buff):
    """
    fully connected nw for encoder
    :param encoder: encoder op
    :return:
    """
    # encoded, inp_buff = encoder_network(inp_shape=inp_shape)
    flat = tf.keras.layers.Flatten(name='flat_1')(encoded)
    dense_1 = tf.keras.layers.Dense(name='dense1', units=128, activation='relu')(flat)
    out = tf.keras.layers.Dense(name='out', units=num_classes, activation='linear')(dense_1)
    return tf.keras.Model(inputs=inp_buff, outputs=out, name='fc_model')


if __name__ == '__main__':
    print("starting.......")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train = pre_process_data(x_train)
    x_test = pre_process_data(x_test)

    print(y_train.shape)
    # y_train = tf.keras.backend.squeeze(y_train, axis=1)
    # y_test = tf.keras.backend.squeeze(y_test, axis=1)

    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    print('train data shape: ', x_train.shape, y_train.shape)
    print('test data shape: ', x_test.shape, y_test.shape)

    """
    hyper-params
    """
    batch_size = 64
    epochs = 30
    num_class = 10
    learning_rate = 0.005
    validation_split = 0.05
    auto_encoder_model = auto_encoder_model(inp_shape=[28, 28, 1])

    auto_encoder_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                               metrics=['accuracy', 'mse'])

    history = auto_encoder_model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs,
                                     validation_split=validation_split)

    print("train scores:")
    print(auto_encoder_model.evaluate(x_train, x_train))
    print("test scores:")
    print(auto_encoder_model.evaluate(x_test, x_test))

    auto_encoder_model.save(filepath='./saved_autoencoder', overwrite=True)




