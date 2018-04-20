#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
import pre

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10+26) #改成我们的分类数目

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) #注意这里的计算方式，所以我们要给0-35

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn():
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {"x": tf.placeholder(shape=[None, 4], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def read_as_array(fn="mnist_sample.png"):
    im = Image.open(fn)
    w, h = im.size
    pix = np.array(im, dtype=np.float32)
    # pix = np.reshape(pix, (1, 28*28))
    pix = pix * np.array([1.0/255])
    pix = pix.astype(np.float32)
    return pix


def save_mnist_sample(eval_data):
    # 把一张测试图片做出png
    ig = eval_data[:1, ]
    ig = ig.reshape((28, 28))
    im = Image.fromarray(ig)
    im = im.point(lambda x: x*255)
    im = im.convert('L')
    im.show()
    im.save("your_file.png")

def c2i(c):
    if ord(c) <= ord('9'):
        return ord(c) - ord('0')
    else:
        return 10+ord(c) - ord('A')

def i2c(i):
    if i >= 10:
        return chr(i-10+ord('A'))
    else:
        return chr(i+ord('0'))

def load_data_set(fn_labels):
    print('loading...  ', fn_labels)
    with open(fn_labels, 'r') as fp_label:
        ll = fp_label.readlines()
        labels = [c2i(l.split(',')[1].strip()) for l in ll ]
        labels = np.array(labels)
        print(labels.shape, labels)
        data = []
        for i, line in enumerate(ll):
            if i%1000==0:
                print('{}/{}'.format(i, len(ll)))

            line = line.strip()
            ww = line.split(',')
            data.append(read_as_array(ww[0]))

        data = np.array(data)
        data = data.reshape((-1,28*28))
        return  data, labels

def load_classfier():
    print('load classfier...')
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model/deyzm_model")

    return mnist_classifier

def yzm_predict(classifier, fn_yzm):
    data = pre.pre_to_mem(fn_yzm)
    if data is not None:
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            num_epochs=1,
            shuffle=False)
        predictions = list(classifier.predict(input_fn=predict_input_fn))
        predicted_classes = [p["classes"] for p in predictions]
        print(predicted_classes)
        predicted_classes = [i2c(i) for i in predicted_classes]
        return ''.join(predicted_classes)
    else:
        return ''

def main(unused_argv):
    # Load training and eval data
    '''
    data format
        (55000, 784) [
        0.         0.         0.         0.         0.         0.
        0.         0.         0.54901963 0.9843138  0.9960785  0.9960785
        0.9960785  0.9960785  0.9960785  0.9960785  0.9960785  0.996078
    label format
        (55000,) [7]
    '''
    train_data, train_labels = load_data_set('train_labels.txt')
    eval_data, eval_labels = load_data_set('test_labels.txt')

    print('train_labels', train_labels.shape, train_labels[:1,])
    print('train_data', train_data.shape, train_data[:1, ])

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model/deyzm_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=4000, #原20000，实际本例到4000，loss就到底了
        hooks=[])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
    tf.app.run()
