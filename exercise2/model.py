import tensorflow as tf
import numpy as np

class cnn_mnist_model() :
    def __init__(self, lr, num_filters, filter_size):

        self.sess = None
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # MNIST images are 28x28 pixels, and have one color channel
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(tf.int64, shape=[None, 1])

        self.batch_size = tf.placeholder(tf.int64)
        self.dataset_size = tf.placeholder(tf.int64)
        dataset_train = tf.data.Dataset.from_tensor_slices((self.x, self.y)).repeat().shuffle(buffer_size=self.dataset_size).batch(self.batch_size)
        dataset_test = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(self.dataset_size)
        iter = tf.data.Iterator.from_structure(dataset_train.output_types,
                                               dataset_train.output_shapes)

        features, labels = iter.get_next()
        self.train_init_op = iter.make_initializer(dataset_train)
        self.test_init_op = iter.make_initializer(dataset_test)

        # Convolutional Layer #1
        # Input Tensor Shape: [batch_size, 28, 28, 1]
        # Output Tensor Shape: [batch_size, 28, 28, num_filters]
        conv1 = tf.layers.conv2d(
            inputs=features,
            filters=num_filters,
            kernel_size=[filter_size, filter_size],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        # Input Tensor Shape: [batch_size, 28, 28, num_filters]
        # Output Tensor Shape: [batch_size, 14, 14, num_filters]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # Output Tensor Shape: [batch_size, 14, 14, num_filters]
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=num_filters,
            kernel_size=[filter_size, filter_size],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        # Output Tensor Shape: [batch_size, 7, 7, num_filters]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * num_filters])

        # Dense Layer
        # Output Tensor Shape: [batch_size, 128]
        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

        # Add dropout operation; 0.6 probability that element will be kept

        # Logits layer
        logits = tf.layers.dense(inputs=dense, units=10)

        self.predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        # Calculate Loss (for both TRAIN and EVAL modes)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(
            loss=self.loss,
            global_step=tf.train.get_global_step())

        # Add evaluation metrics (for EVAL mode)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(labels), self.predictions["classes"]), tf.float32))

        self.init_op = tf.global_variables_initializer()
        self.local_init_op = tf.local_variables_initializer()

        print("starting TF session and initializing variables like weights etc")
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.sess.run(self.local_init_op)
        print("Done")

    def __del__(self):
        if (self.sess is not None):
            print("Shutting down Tensorflow session in model destructor")
            self.sess.close()

    def train(self, x_train, y_train, x_valid, y_valid, batch_size, num_epochs):

        learning_curve = []
        for i in range(num_epochs):
            print("Starting Epoch {}".format(i))
            self.sess.run(self.train_init_op, feed_dict={self.x: x_train, self.y: y_train, self.batch_size: batch_size, self.dataset_size: x_train.shape[0]})
            n_batches = x_train.shape[0] // batch_size
            for _ in range(n_batches):
                self.sess.run(self.train_op)

            loss, accuracy = self.test_on_batch(x_valid, y_valid)
            learning_curve.append((loss, accuracy))
            print("Validation loss {:.4f}, accuracy: {:4f}".format(loss, accuracy))
            print("Finished Epoch {}".format(i))
            print()

        return learning_curve


    def test_on_batch(self, x_batch, y_batch):
        self.sess.run(self.test_init_op, feed_dict={self.x: x_batch, self.y: y_batch, self.batch_size: x_batch.shape[0], self.dataset_size: x_batch.shape[0]})
        return self.sess.run((self.loss, self.accuracy))
