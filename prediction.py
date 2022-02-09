
from __future__ import division, print_function, unicode_literals #To support both Python 2 and Python 3
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt #To plot pretty figures
import cv2
# import cv2

import tensorflow as tf
print(tf.__version__)

from tensorflow.python.framework import ops 
ops.reset_default_graph()

import numpy as np
#from numpy import asarray

np.random.seed(42)
tf.random.set_seed(45)
from PIL import Image, ImageFilter

def predict_image(preprocessed_digits):

    def imageprepare(argv):
        """
        This function returns the pixel values.
        The imput is a png file location.
        """
        im = Image.open(argv).convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

        if width > height:  # check which dimension is bigger
            # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
                # resize and sharpen
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
                # resize and sharpen
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        # newImage.save("sample.png

        tv = list(newImage.getdata())  # get pixel values

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        print(tva)
        return tva
    #####################################################################
    #Load MNIST data

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
    caps1_n_maps = 32
    caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
    caps1_n_dims = 8
    conv1_params = {
        "filters": 256,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",
        "activation": tf.nn.relu,
    }

    conv2_params = {
        "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
        "kernel_size": 9,
        "strides": 2,
        "padding": "valid",
        "activation": tf.nn.relu
    } 

    conv1 = tf.compat.v1.layers.conv2d(X, name="conv1", **conv1_params)
    conv2 = tf.compat.v1.layers.conv2d(conv1, name="conv2", **conv2_params)

    caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                        name="caps1_raw")

    def squash(s, axis=-1, epsilon=1e-7, name=None):
        with tf.name_scope(name):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                        keepdims=True)
            safe_norm = tf.sqrt(squared_norm + epsilon)
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = s / safe_norm
            return squash_factor * unit_vector

    caps1_output = squash(caps1_raw, name="caps1_output")


    caps2_n_caps = 10
    caps2_n_dims = 16

    init_sigma = 0.1

    W_init = tf.random.normal(
        shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
        stddev=init_sigma, dtype=tf.float32, name="W_init")
    W = tf.Variable(W_init, name="W")

    batch_size = tf.shape(X)[0]
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled") 
    caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                        name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                    name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                                name="caps1_output_tiled")

    print(W_tiled) #shape of the first array

    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                name="caps2_predicted")

    print(caps2_predicted)

    #Routing by agreement
    raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                        dtype=np.float32, name="raw_weights")

    routing_weights = tf.nn.softmax(raw_weights,name="routing_weights")

    weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                    name="weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,
                                name="weighted_sum")

    caps2_output_round_1 = squash(weighted_sum, axis=-2,
                                name="caps2_output_round_1")
    print(caps2_output_round_1)

    print(caps2_predicted)

    print(caps2_output_round_1)

    caps2_output_round_1_tiled = tf.tile(
        caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
        name="caps2_output_round_1_tiled")

    agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                        transpose_a=True, name="agreement")

    raw_weights_round_2 = tf.add(raw_weights, agreement,
                                name="raw_weights_round_2")

    routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                            name="routing_weights_round_2")
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                            caps2_predicted,
                                            name="weighted_predictions_round_2")
    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                        axis=1, keepdims=True,
                                        name="weighted_sum_round_2")
    caps2_output_round_2 = squash(weighted_sum_round_2,
                                axis=-2,
                                name="caps2_output_round_2")

    caps2_output = caps2_output_round_2

    def condition(input, counter):
        return tf.less(counter, 100)

    def loop_body(input, counter):
        output = tf.add(input, tf.square(counter))
        return output, tf.add(counter, 1)

    with tf.name_scope("compute_sum_of_squares"):
        counter = tf.constant(1)
        sum_of_squares = tf.constant(0)

        result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])
        

    with tf.compat.v1.Session() as sess:
        print(sess.run(result))

    print(sum([i**2 for i in range(1, 100 + 1)]))

    def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
        with tf.name_scope(name):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                        keepdims=keep_dims)
            return tf.sqrt(squared_norm + epsilon)

    y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
    y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
    print(y_proba_argmax)

    y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
    print(y_pred)
    y = tf.compat.v1.placeholder(shape=[None], dtype=tf.int64, name="y")
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    T = tf.one_hot(y, depth=caps2_n_caps, name="T")

    with tf.compat.v1.Session():
        print(T.eval(feed_dict={y: np.array([0, 1, 2, 3, 9])}))

    print(caps2_output)
    caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                                name="caps2_output_norm")

    present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                name="present_error_raw")
    present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                            name="present_error")

    absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                name="absent_error_raw")
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                            name="absent_error")

    L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
            name="L")
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    mask_with_labels = tf.compat.v1.placeholder_with_default(False, shape=(),
                                                name="mask_with_labels")

    reconstruction_targets = tf.cond(mask_with_labels, # condition
                                    lambda: y,        # if True
                                    lambda: y_pred,   # if False
                                    name="reconstruction_targets")

    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                    depth=caps2_n_caps,
                                    name="reconstruction_mask")

    print(reconstruction_mask)

    print(caps2_output)

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
        name="reconstruction_mask_reshaped")

    caps2_output_masked = tf.multiply(
        caps2_output, reconstruction_mask_reshaped,
        name="caps2_output_masked")

    print(caps2_output_masked)

    decoder_input = tf.reshape(caps2_output_masked,
                            [-1, caps2_n_caps * caps2_n_dims],
                            name="decoder_input")

    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = 28 * 28

    with tf.name_scope("decoder"):
        hidden1 = tf.compat.v1.layers.dense(decoder_input, n_hidden1,
                                activation=tf.nn.relu,
                                name="hidden1")
        hidden2 = tf.compat.v1.layers.dense(hidden1, n_hidden2,
                                activation=tf.nn.relu,
                                name="hidden2")
        decoder_output = tf.compat.v1.layers.dense(hidden2, n_output,
                                        activation=tf.nn.sigmoid,
                                        name="decoder_output")
    #Reconstruction Loss

    X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output,
                                name="squared_difference")
    reconstruction_loss = tf.reduce_mean(squared_difference,
                                        name="reconstruction_loss")

    alpha = 0.0005

    loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

    correct = tf.equal(y, y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    optimizer = tf.compat.v1.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.08, random_state=42)

    n_epochs = 1
    batch_size = 50
    restore_checkpoint = True

    n_iterations_per_epoch = len(X_train) // batch_size
    n_iterations_validation = len(X_val) // batch_size
    best_loss_val = np.infty

    checkpoint_path = "./my_capsule_network"

    sample_images = []
    i= 0


    y_pred_value_array = []
    digit_array = []
    for digit in preprocessed_digits:
        sample_image = digit.reshape([-1, 28, 28, 1]).astype('float32')
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, checkpoint_path)
            caps2_output_value, decoder_output_value, y_pred_value = sess.run(
                    [caps2_output, decoder_output, y_pred],
                    feed_dict={X: sample_image,
                            y: np.array([], dtype=np.int64)})
        
            

            sample_image = sample_image.reshape(-1, 28, 28)
            reconstructions = decoder_output_value.reshape([-1, 28, 28])

            plt.imshow(digit, cmap="binary")
            plt.title("Predicted:" + str(y_pred_value))
            plt.show()
            
            
            y_pred_value_array.append(y_pred_value)
    print(y_pred_value_array) 
    return y_pred_value_array