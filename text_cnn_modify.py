import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, slot_dim, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.slot = tf.placeholder(tf.float32, [None, slot_dim], name="slot")
        self.zero_num = tf.placeholder(tf.float32, name = "zero_num")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.labels = tf.argmax(self.input_y, 1, name="labels")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # print(self.h_pool)
        self.h_pool_slot = tf.reshape(self.h_pool, [-1, num_filters_total], name="pool")
        self.h_pool_slot = tf.layers.dense(inputs=self.h_pool_slot, units=18, activation=tf.nn.relu)
        self.h_pool_flat = tf.concat([self.h_pool_slot, self.slot], 1)
        self.h_pool_flat = tf.layers.batch_normalization(self.h_pool_flat, axis=1)
        self.slot_dense_0 = tf.layers.dense(inputs=self.h_pool_slot, units=64, activation=tf.nn.relu, trainable=True,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

        # self.slot_dense_0_1 = tf.layers.dense(inputs=self.h_pool_flat, units=64, activation=tf.nn.relu, trainable=True,
        #                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

        # self.slot_dense_1 = tf.layers.dense(inputs=self.slot_dense_0, units=128, activation=tf.nn.relu, trainable=True,
        #                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        # self.slot_dense_2 = tf.layers.dense(inputs=self.slot_dense_1, units=256, activation=tf.nn.relu, trainable=True,
        #                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        self.slot_dense_3 = tf.layers.dense(inputs=self.slot_dense_0, units=num_filters_total, activation=None, trainable=True,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        # self.slot_dense_4 = tf.layers.dense(inputs=self.slot_dense_3, units=3, activation=tf.nn.relu)

        self.h_pool_flat = self.slot_dense_3
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # with tf.name_scope("dense"):
            # self.slot_dense = tf.layers.dense(inputs=self.slot, units=16, activation=tf.nn.relu())
            # self.scores_slot = tf.concat([self.scores, self.slot], 1)
            # self.slot_dense_1 = tf.layers.dense(inputs=self.scores_slot, units=1024, activation=tf.nn.relu)
            # self.slot_dense_2 = tf.layers.dense(inputs=self.slot_dense_1, units=512, activation=tf.nn.relu)
            # self.slot_dense_3 = tf.layers.dense(inputs=self.slot_dense_2, units=256, activation=tf.nn.relu)
            # self.slot_dense_4 = tf.layers.dense(inputs=self.slot_dense_3, units=3, activation=tf.nn.relu)
            # self.predictions = tf.argmax(self.slot_dense_4, 1, name="predictions")
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")