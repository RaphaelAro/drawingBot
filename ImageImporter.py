from PIL import Image
from misc import print_progress

import numpy as np
import os
import time

# has to be x , x
IMAGESIZE = 250, 250

PATH_CALTECH = "Caltech256"
PATH_LFW = "lfw"

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
    return image

def showImage(image):
    PILImage = Image.fromarray(np.uint8(np.round(image * 255)))
    PILImage.show()
    return

def loadDataset(path):
    #Determine number of images
    count = 0
    for dir in os.listdir(path):
        for file in os.listdir(path + "/" + dir):
            if file.endswith(".jpg"):
                count = count + 1
    print("loading dataset from " + path + " (%d images) ..." % count)

    images = np.zeros([count, IMAGESIZE[0], IMAGESIZE[1]])

    start = int(round(time.time()))

    #import all images to array
    i = 0
    for dir in os.listdir(path):
        for file in os.listdir(path + "/" + dir):
            if file.endswith(".jpg"):
                imgPath = os.path.join(path, dir, file)
                images[i] = loadBWPic(imgPath)
                print_progress(i, count, start_time_seconds=start)
                i = i + 1

    print("\n%d files loaded" % i)
    return images

caltech = loadDataset(PATH_CALTECH)
lfw = loadDataset(PATH_LFW)

dataset = np.concatenate((caltech, lfw))

print(dataset.shape)

showImage(caltech[400])

#showImage(image)
