import os
import unittest.mock
import numpy as np
import tensorflow as tf
# Modify the line below based on your last name

from Gunasekharan_03_01 import train_nn_keras, confusion_matrix


@unittest.mock.patch("tensorflow.keras.utils.set_random_seed")
def test_seed_called(random_seed):
    # make sure the first thing your function does is calls set_random_seed!
    X_train = np.zeros((100, 28, 28, 1))
    Y_train = np.zeros((100, 10))
    X_test = np.zeros((100, 28, 28, 1))
    Y_test = np.zeros((100, 10))
    train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=0, batch_size=1)

    random_seed.assert_called_once_with(5368), "This makes sure you didn't remove the random seed function call. No credit for passing this test!"

def test_model_architecture():
    # this tests that your model architecture is correct
    # please adhere to this exactly

    # make dummy data for testing purposes
    X_train = np.zeros((100, 28, 28, 1))
    Y_train = np.zeros((100, 10))
    X_test = np.zeros((100, 28, 28, 1))
    Y_test = np.zeros((100, 10))

    # build model
    model, history, cm, Y_pred = train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=0, batch_size=1)

    # BEGIN verifying architecture parameters
    assert len(model.layers) == 10
    assert isinstance(model.layers[0], tf.keras.layers.Conv2D)
    assert model.layers[0].filters == 8
    assert model.layers[0].kernel_size == (3, 3)
    assert model.layers[0].strides == (1, 1)
    assert model.layers[0].padding == 'same'
    assert model.layers[0].activation._name_ == 'relu'
    assert model.layers[0].input_shape == (None, 28, 28, 1)
    assert model.layers[0].kernel_regularizer._class.name_ == 'L2'
    assert abs(float(model.layers[0].kernel_regularizer.l2) - 0.0001) < 1e-8

    assert isinstance(model.layers[1], tf.keras.layers.Conv2D)
    assert model.layers[1].filters == 16
    assert model.layers[1].kernel_size == (3, 3)
    assert model.layers[1].strides == (1, 1)
    assert model.layers[1].padding == 'same'
    assert model.layers[1].activation._name_ == 'relu'


    assert isinstance(model.layers[2], tf.keras.layers.MaxPooling2D)
    assert model.layers[2].pool_size == (2, 2)
    assert model.layers[2].strides == (2, 2)
    assert model.layers[2].padding == 'valid'


    assert isinstance(model.layers[3], tf.keras.layers.Conv2D)
    assert model.layers[3].filters == 32
    assert model.layers[3].kernel_size == (3, 3)
    assert model.layers[3].strides == (1, 1)
    assert model.layers[3].padding == 'same'
    assert model.layers[3].activation._name_ == 'relu'


    assert isinstance(model.layers[4], tf.keras.layers.Conv2D)
    assert model.layers[4].filters == 64
    assert model.layers[4].kernel_size == (3, 3)
    assert model.layers[4].strides == (1, 1)
    assert model.layers[4].padding == 'same'
    assert model.layers[4].activation._name_ == 'relu'

    assert isinstance(model.layers[5], tf.keras.layers.MaxPooling2D)
    assert model.layers[5].pool_size == (2, 2)
    assert model.layers[5].strides == (2, 2)
    assert model.layers[5].padding == 'valid'

    assert isinstance(model.layers[6], tf.keras.layers.Flatten)

    assert isinstance(model.layers[7], tf.keras.layers.Dense)
    assert model.layers[7].units == 512
    assert model.layers[7].activation._name_ == 'relu'
    assert model.layers[7].kernel_regularizer._class.name_ == 'L2'
    assert abs(float(model.layers[7].kernel_regularizer.l2) - 0.0001) < 1e-8

    assert isinstance(model.layers[8], tf.keras.layers.Dense)
    assert model.layers[8].units == 10
    assert model.layers[8].activation._name_ == 'linear'
    assert model.layers[8].kernel_regularizer._class.name_ == 'L2'
    assert abs(float(model.layers[8].kernel_regularizer.l2) - 0.0001) < 1e-8

    assert isinstance(model.layers[9], tf.keras.layers.Activation)
    assert model.layers[9].activation._name_ == 'softmax'


def test_save_model():
    # remove the file model.h5 if it exists
    import os
    if os.path.exists('model.h5'):
        os.remove('model.h5')

    # make dummy data for testing purposes
    X_train = np.zeros((100, 28, 28, 1))
    Y_train = np.zeros((100, 10))
    X_test = np.zeros((100, 28, 28, 1))
    Y_test = np.zeros((100, 10))

    # build model
    model, history, cm, Y_pred = train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=0, batch_size=1)
    assert os.path.exists('model.h5')
    model_loaded = tf.keras.models.load_model('model.h5')
    assert model_loaded != None
    # if the model loads without error, we'll assume it's correct for this test (other tests check it)

def test_confusion_matrix():
    # verify confusion matrix using tf.math.confusion_matrix
    # make dummy data for testing purposes
    Y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cm = confusion_matrix(Y_true, Y_pred)
    assert cm.shape == (10, 10)
    ground_truth = tf.math.confusion_matrix(Y_true, Y_pred)
    assert np.all(cm == ground_truth), f"Your confusion matrix is incorrect.  It should be:\n{ground_truth} not \n{cm}"

    # make more dummy data for testing purposes
    Y_true = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Y_pred = np.array([1, 1, 1, 1, 1, 0, 1, 3, 4, 6, 8, 8, 9])
    cm = confusion_matrix(Y_true, Y_pred)
    assert cm.shape == (10, 10)
    ground_truth = tf.math.confusion_matrix(Y_true, Y_pred)
    assert np.all(cm == ground_truth), f"Your confusion matrix is incorrect.  It should be:\n{ground_truth} not \n{cm}"

    # these tests make sure you've written the function yourself
    with unittest.mock.patch('tensorflow.math.confusion_matrix') as mock_confusion_matrix:
        confusion_matrix(Y_true, Y_pred)
        mock_confusion_matrix.assert_not_called(), "DO NOT use tf.math.confusion_matrix!"

    # with unittest.mock.patch('sklearn.metrics.confusion_matrix') as mock_confusion_matrix:
    #     confusion_matrix(Y_true, Y_pred)
    #     mock_confusion_matrix.assert_not_called(), "DO NOT use sklearn.metrics.confusion_matrix!"

@unittest.mock.patch("matplotlib.pyplot.matshow")
@unittest.mock.patch("matplotlib.axes._axes.Axes.matshow")
def test_confusion_matrix_plot(plt_matshow, ax_matshow):
    # delete confusion_matrix.png if it exists
    if os.path.exists('confusion_matrix.png'):
        os.remove('confusion_matrix.png')

    # make dummy data for testing purposes
    X_train = np.zeros((100, 28, 28, 1))
    Y_train = np.zeros((100, 10))
    X_test = np.zeros((100, 28, 28, 1))
    Y_test = np.zeros((100, 10))

    # train 1 epoch
    model, history, cm, Y_pred = train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=1)
    assert plt_matshow.called or ax_matshow.called, "Did you call matshow? Please use either plt.matshow() or axes.matshow() (if you're using subplots)"
    assert os.path.exists('confusion_matrix.png'), "Did you save the confusion matrix to confusion_matrix.png using savefig?"

def test_accuracy_on_mnist():
    # this is the most realistic test, using the MNIST dataset
    # you can paste the body of this function into your main program in an if _name_ == '_main_' block to test it

    # load mnist data
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    # normalize data
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)
    # reshape data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # one-hot encode labels
    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test, 10)


    # # select the first 100 samples for training
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    # # select the first 100 samples for testing
    X_test = X_test[:100]
    Y_test = Y_test[:100]

    # train model for 5 epochs with batch size 5
    model, history, cm, Y_pred = train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=5, batch_size=5)

    assert history.history['accuracy'][-1] > 0.8, "This model should get at least 80% accuracy on such a small sample of the training set"

    assert history.history['val_accuracy'][-1] > 0.8, "This model should get at least 80% accuracy on the validation set with just 100 samples"
    # note: this is the validation set that keras automatically generates from the train set, not the test set


def test_ypred():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    # normalize data
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)
    # reshape data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # one-hot encode labels
    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test, 10)

    # # select the first 100 samples for training
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    # # select the first 100 samples for testing
    X_test = X_test[:100]
    Y_test = Y_test[:100]

    # train
    model, history, cm, Y_pred = train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=2, batch_size=5)

    ground_truth = np.array([9, 3, 1, 0, 4, 1, 4, 9, 4, 9, 0, 4, 9, 0, 1, 3, 4, 3, 3, 4, 9, 3,
       4, 3, 1, 0, 9, 4, 0, 1, 3, 1, 3, 4, 3, 3, 3, 1, 3, 1, 1, 9, 4, 1,
       3, 3, 1, 1, 4, 4, 6, 3, 9, 9, 4, 0, 4, 1, 4, 1, 3, 3, 1, 3, 4, 1,
       1, 4, 3, 0, 3, 0, 3, 3, 1, 9, 3, 1, 1, 0, 9, 6, 3, 9, 4, 4, 3, 3,
       6, 1, 3, 6, 1, 3, 1, 4, 1, 3, 6, 4], dtype=np.int64)
    # print(f"{Y_pred = !r}")

    assert Y_pred.shape == (100,), "Y_pred should be a 1-D array of length 100"
    assert np.allclose(Y_pred, ground_truth), "Y_pred does not match the expected output"