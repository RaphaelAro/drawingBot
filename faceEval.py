import tensorflow as tf
import numpy as np

from misc import *
from ImageImporter import *

#Path where to save trained model
DIR_ESTIMATOR = "estimator"
DIR_MODEL = "model"

BATCH_SIZE = 100
TRAINING_ITERATIONS = 20000


def cnnModel(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 128, 128, 1])
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    #Convolutional Layer #2 and PoolingLayer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=5,
        padding="same"
        ,activation=tf.nn.relu
    )
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool4, [-1, 8 * 8 * 256])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    prediction_output = tf.estimator.export.PredictOutput({"classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")})

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output})

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    # Load training and eval data
    dataset = trainingData()
    train_data = np.asarray(dataset[0], dtype=np.float32)
    train_labels = np.asarray(dataset[1], dtype=np.int32)
    # devide sets
    devide = int(round(4 * np.shape(train_data)[0] / 5))
    eval_data = train_data[devide:]
    eval_labels = train_labels[devide:]
    train_data = train_data[0:devide]
    train_labels = train_labels[0:devide]



    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnnModel, model_dir=DIR_ESTIMATOR)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True
    )

    classifier.train(
        input_fn=train_input_fn,
        steps=TRAINING_ITERATIONS,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


    # Save model
    def serving_input_receiver_fn():
        feature_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 16384])
        return tf.estimator.export.ServingInputReceiver({'x': feature_tensor}, {'x': feature_tensor})

    classifier.export_savedmodel(DIR_MODEL, serving_input_receiver_fn)
    print("Saved model to '%s'..." % DIR_MODEL)


if __name__ == "__main__":
    tf.app.run()
