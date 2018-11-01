from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels

def one_un_hot(hot_labels):
    labels = np.argmax(hot_labels, axis=-1)
    labels = np.squeeze(labels)
    return labels



def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size = 3):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_filters, (filter_size, filter_size), (1, 1), padding="same"),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(num_filters, (filter_size, filter_size), (1, 1), padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D((2, 2), 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    sgd = tf.keras.optimizers.SGD(lr=lr)
    model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    learning_curve = []
    for _ in range(num_epochs):

        model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
        res = model.evaluate(x_valid, y_valid)
        learning_curve.append(res)

    return learning_curve, model  # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    print("Evaluating model on test set")
    res = model.evaluate(x_valid, y_valid)
    test_error = 1.0 - res[1]
    print("Test loss: {}\n Test error {}".format(res[0], res[1]))
    return test_error

def exercise_2(x_train, y_train, x_valid, y_valid,):
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    curves = []
    for lr in learning_rates:
        curve, _ = train_and_validate(x_train, y_train, x_valid, y_valid, 12, lr, 16, 128)
        curves.append(curve)

    import matplotlib.pyplot as plt
    x = [i for i in range(len(curves[0]))]
    colors = ['g', 'r', 'b', 'c']
    for color, curve, lr in zip(colors, curves, learning_rates):
        losses = [x[0] for x in curve]
        plt.plot(x, losses, '{}--'.format(color), label='lr={}'.format(lr))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Validation Loss on MNIST for different learning Rates')
    plt.savefig('Ex2.eps')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--exercise_two", action='store_true',
                        help="execute exercise two and create plots. Other arguments are ignored")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    if args.exercise_two:
        exercise_2(x_train, y_train, x_valid, y_valid)
        import sys
        sys.exit()


    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size)

    test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
