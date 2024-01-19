# Gunasekharan, Jayasurya
# 1002_060_473
# 2023_10_29
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def confusion_matrix(y_true, y_pred, n_classes=10):
    # Compute the confusion matrix for a set of predictions
    # the shape of the confusion matrix should be (n_classes, n_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    # The (i, j)th entry should be the number of times an example with true label i was predicted label j
    # Cast y_true and y_pred to integers to ensure they are used as valid indices
    # Do not use any libraries to use this function (e.g. sklearn.metrics.confusion_matrix, or tensorflow.math.confusion_matrix, ..)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    # Only use numpy.
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        confusion[true_label][pred_label] += 1

    return confusion

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4, validation_split=0.2):
    # Build the neural network model
    # Train a convolutional neural network using Keras.
    # X_train: float 32 numpy array [number_of_training_samples, 28, 28, 1]
    # print("X_train", X_train)
    # print("X_train.shape", X_train.shape)
    # Y_train: float 32 numpy array [number_of_training_samples, 10]  (one hot format)
    # print("Y_train", Y_train)
    # print("Y_train.shape", Y_train.shape)
    # X_test: float 32 numpy array [number_of_test_samples, 28, 28, 1]
    # print("X_test", X_test)
    # print("X_test.shape", X_test.shape)
    # Y_test: float 32 numpy array [number_of_test_samples, 10]  (one hot format)
    # print("Y_test", Y_test)
    # print("Y_test.shape", Y_test.shape)
    # Assume that the data has been preprocessed (normalized). You do not need to normalize the data.

    # Split the training data into training and validation sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=validation_split, random_state=42)

    # The neural network should have this exact architecture (it's okay to hardcode)
    # Number of Parameters = (filter_height * filter_width * input_channels + 1) * number_of_features
    model = tf.keras.models.Sequential()
    # - Convolutional layer with 8 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001), input_shape=(28, 28, 1)))

    # - Convolutional layer with 16 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001)))

    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # - Convolutional layer with 32 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001)))

    # - Convolutional layer with 64 filters, kernel size 3 by 3 , stride 1 by 1, padding 'same', and ReLU activation
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0001)))

    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # - Flatten layer
    model.add(Flatten())

    # - Dense layer with 512 units and ReLU activation
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0001)))

    # - Dense layer with 10 units with linear activation
    model.add(Dense(10, activation='linear', kernel_regularizer=l2(0.0001)))

    # - a softmax layer
    model.add(Activation('softmax'))


    # Summary to see the architecture of the model
    print(model.summary())

    # The neural network should be trained using the Adam optimizer with default parameters
    # The loss function should be categorical cross-entropy.
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # The number of epochs should be given by the 'epochs' parameter.
    # The batch size should be given by the 'batch_size' parameter.
    # All layers that have weights should have L2 regularization with a regularization strength of 0.0001 (only use kernel regularizer)
    # All other parameters should use keras defaults'
    # Train the model
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Compute the confusion matrix for the test set
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    # print("Y_pred", Y_pred)
    # print("Y_pred shape", Y_pred.shape)

    # You should compute the confusion matrix on the test set and return it as a numpy array.
    cm = confusion_matrix(Y_test, Y_pred)

    # You should plot the confusion matrix using the matplotlib function matshow (as heat map) and save it to 'confusion_matrix.png'
    plt.matshow(cm)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('confusion_matrix.png')

    # Show the plot (optional)
    # plt.show()

    # You should save the keras model to a file called 'model.h5' (do not submit this file). When we test run your program we will check "model.h5"
    model.save('model.h5')

    # Your program should run uninterrupted and require no user input (including closing figures, etc).
    # You will return a list with the following items in the order specified below:
    # - the trained model
    # - the training history (the result of model.fit as an object)
    # - the confusion matrix as numpy array
    # - the output of the model on the test set (the result of model.predict) as numpy array
    tf.keras.utils.set_random_seed(5368)  # do not remove this line
    return [model, history, cm, Y_pred]
