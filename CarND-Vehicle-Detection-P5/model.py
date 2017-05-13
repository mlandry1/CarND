from keras.models import Sequential
from keras.layers import Dropout, Lambda
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D


def get_model(input_shape=(64, 64, 3)):
    model = Sequential()
    # Center and normalize the data
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(50, (3, 3), activation='relu', padding="same", strides=(1, 1)))
    model.add(Dropout(0.5))
    model.add(Conv2D(50, (3, 3), activation='relu', padding="same", strides=(1, 1)))
    model.add(Dropout(0.5))
    model.add(Conv2D(50, (3, 3), activation='relu', padding="same", strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))
    model.add(Conv2D(90, (8, 8), activation="relu", strides=(1, 1)))
    model.add(Dropout(0.5))
    # This is like a 1 neuron dense layer with tanh [-1, 1]
    model.add(Conv2D(1, (1, 1), activation="tanh", strides=(1, 1)))

    return model
