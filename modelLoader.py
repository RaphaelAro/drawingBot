import numpy as np
import tensorflow as tf

from ImageImporter import *

FULL_DIR_MODEL = "model/1518986160"

def predictPath(path):
    image = np.asarray(loadBWPic(path), dtype=np.float32)

    sess = tf.Session()
    # load the saved model
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FULL_DIR_MODEL)
    predictor = tf.contrib.predictor.from_saved_model(FULL_DIR_MODEL)

    output_dict = predictor({"x": [image]})
    sess.close()

    return output_dict['probabilities'][0][1]

def predictImage(image):
    sess = tf.Session()
    # load the saved model
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FULL_DIR_MODEL)
    predictor = tf.contrib.predictor.from_saved_model(FULL_DIR_MODEL)

    output_dict = predictor({"x": [image]})
    sess.close()

    return output_dict['probabilities'][0][1]

def main(argv):
    files = ["test1.jpg", "test2.jpg"]
    prediction = predictPath(files[0])
    print("Probability for face: %s (%s)" % (prediction, files[0]))

    prediction = predictPath(files[1])
    print("Probability for face: %s (%s)" % (prediction, files[1]))


if __name__ == "__main__":
    tf.app.run()
