from PIL import Image
from misc import print_progress

import numpy as np
import os
import time

# has to be x , x
IMAGESIZE = 128, 128

PATH_CALTECH = "Caltech256LeftOut"
PATH_LFW = "Caltech256LeftOut"

def loadBWPic(path):
    image = Image.open(path).convert('LA')
    width, height = image.size
    #cut to square
    if width != height:
        if width > height:
            diff = round((width - height) / 2)
            image = image.crop((diff, 0, height + diff, height))
        else:
            diff = round((height - width) / 2)
            image = image.crop((0, diff, width, width + diff))

    #resize
    if image.size != IMAGESIZE:
        image = image.resize(IMAGESIZE, Image.ANTIALIAS)

    image = np.asarray(image)
    image = np.array(image[:,:,0]) / 255.0
    image = image.flatten()
    return image

def showImage(image):
    image = np.resize(image, IMAGESIZE)
    PILImage = Image.fromarray(np.uint8(np.round(image * 255)))
    PILImage.show()
    return

def saveImage(image, path):
    PILImage = Image.fromarray(np.uint8(np.round(image * 255)))
    PILImage.save(path)
    return

def loadDataset(path, label):
    #Determine number of images
    count = 0
    for dir in os.listdir(path):
        for file in os.listdir(path + "/" + dir):
            if file.endswith(".jpg"):
                count = count + 1
    print("loading dataset from " + path + " (%d images) ..." % count)

    images = np.zeros([count, IMAGESIZE[0] * IMAGESIZE[1]])
    labels = np.empty(count)
    labels.fill(label)

    start = int(round(time.time()))

    #import all images to array
    i = 0
    for dir in os.listdir(path):
        for file in os.listdir(path + "/" + dir):
            if file.endswith(".jpg"):
                imgPath = os.path.join(path, dir, file)
                images[i] = loadBWPic(imgPath)
                i = i + 1
                print_progress(i, count, start_time_seconds=start)

    return (images, labels)

def trainingData():
    caltech = loadDataset(PATH_CALTECH, 0)
    lfw = loadDataset(PATH_LFW, 1)

    print("concatenating datasets...")
    dataset = np.concatenate((caltech[0], lfw[0]))
    print("concatenating labels...")
    labels = np.concatenate((caltech[1], lfw[1]))

    print("shuffle...")
    seed = int(round(time.time()))

    #Use same seed for both arrays!
    np.random.seed(seed)
    np.random.shuffle(dataset)
    np.random.seed(seed)
    np.random.shuffle(labels)

    return (dataset, labels)


if __name__ == "__main__":
  test = trainingData()
  data = test[0]
  labels = test[1]
  showImage(data[0])
  print(labels[0])
