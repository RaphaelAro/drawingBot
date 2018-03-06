import numpy as np
import tensorflow as tf

from ImageImporter import *
from modelLoader import *

IMLENGTH = IMAGESIZE[0] * IMAGESIZE[1]

def randomChange(image, maxChange):
    add = np.random.rand(IMLENGTH)
    for i in range(0, IMLENGTH):
        num = (add[i] - 0.5)/maxChange
        if num > 1:
            num = 1
        if num < 0:
            num = 0
        image[i] = num

    return image

def randomChangePatch(image, maxChange, x, y, patchSize):
    image = image.reshape(IMAGESIZE[0], IMAGESIZE[1])
    size = patchSize ** 2
    halfPatchSize = int(patchSize / 2)
    add = np.random.rand(size)
    i = 0
    for patch_x in range(max(0, x - halfPatchSize), min(IMAGESIZE[0], x + patchSize - halfPatchSize)):
        for patch_y in range(max(0, y - halfPatchSize), min(IMAGESIZE[1], y + patchSize - halfPatchSize)):
            num = (add[i] - 0.5) / maxChange
            if num > 1:
                num = 1
            if num < 0:
                num = 0
            image[patch_x, patch_y] = num
            i += 1

    return image.reshape(IMLENGTH)

def randomImage():
    return np.random.rand(IMLENGTH)

def randomOptimizer():
    image = np.zeros(IMLENGTH)
    image.fill(0.5)

    sess = tf.Session()
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FULL_DIR_MODEL)
    predictor = tf.contrib.predictor.from_saved_model(FULL_DIR_MODEL)

    def predict(image):
        output_dict = predictor({"x": [image]})
        return output_dict['probabilities'][0][1]


    prob = predict(image)

    iterations = 1000000
    for i in range(0, iterations):
        changedImage = randomChange(image, 1)
        prob2 = predict(changedImage)
        if prob2 > prob:
            image = changedImage
            prob = prob2

        if prob > 0.5:
            print("Step %s done. Probability: %s" % (i, prob))
            break

        if i % 100 == 0:
            print("Step %s done. Probability: %s" % (i, prob))

    showImage(image)


def randomOptimizerPatches():
    image = np.zeros(IMLENGTH)
    image.fill(0.5)

    sess = tf.Session()
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FULL_DIR_MODEL)
    predictor = tf.contrib.predictor.from_saved_model(FULL_DIR_MODEL)

    def predict(image):
        output_dict = predictor({"x": [image]})
        return output_dict['probabilities'][0][1]


    prob = predict(image)

    iterations = 100000
    for i in range(0, iterations):
        changedImage = randomChange(image, 1)
        prob2 = predict(changedImage)
        if prob2 > prob:
            image = changedImage
            prob = prob2

        if prob > 0.30:
            print("Step %s done. Probability: %s" % (i, prob))
            break

        if i % 100 == 0:
            print("Step %s done. Probability: %s" % (i, prob))

    iterations = 2
    for i in range(1, iterations+1):
        patchSize = 7
        for x in range(0, IMAGESIZE[0]):
            for y in range(0, IMAGESIZE[1]):
                changedImage = randomChangePatch(image, 1, x, y, patchSize)
                prob2 = predict(changedImage)
                if prob2 > prob:
                    image = changedImage
                    prob = prob2
                    print("Probability: %s" % (prob))

        print("Step %s done." % (i))

    showImage(image)


def main(argv):
    randomOptimizerPatches()

if __name__ == "__main__":
   tf.app.run()
