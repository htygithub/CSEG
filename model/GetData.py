import os
import random

import numpy as np

from scipy.io import loadmat

def conv2arr_and_transpose(input, reshape):
    temp = np.asarray(input)
    temp = np.transpose(temp, (0, 3, 1, 2))
    output = temp.reshape(reshape)
    print (output.shape)
    return output

def Load_Data(mat):
    Group_quantity = 40
    Reserved_quantity = 10

    training_images = []
    training_labels = []
    test_images = []
    test_labels = []

    print("loading images")

    Data = loadmat(mat)
    images = Data['images']
    labels = Data['labels']
    images_list = list(images)
    labels_list = list(labels)

    for n in range(5):
        training_images = training_images + images_list[Reserved_quantity + Group_quantity*n : Group_quantity * (n+1)]
        training_labels = training_labels + labels_list[Reserved_quantity + Group_quantity*n : Group_quantity * (n+1)]
        test_images = test_images + images_list[Group_quantity*n : Reserved_quantity + Group_quantity*n]
        test_labels = test_labels + labels_list[Group_quantity*n : Reserved_quantity + Group_quantity*n]

    training_images = conv2arr_and_transpose(training_images, (1500,256,256))
    training_labels = conv2arr_and_transpose(training_labels, (1500,256,256))
    test_images = conv2arr_and_transpose(test_images, (500,256,256))
    test_labels = conv2arr_and_transpose(test_labels, (500,256,256))

    print("finished loading images")

    return training_images, training_labels, test_images, test_labels

class GetData():
    def __init__(self, image, label):
        self.source_list = []
        self.examples = image.shape[0]
        print("Number of examples found: ", self.examples)
        self.images = image[...,None]
        #self.labels = label[...,None]
        self.labels = label

    def next_batch(self, batch_size):
        if len(self.source_list) < batch_size:
            new_source = list(range(self.examples))
            random.shuffle(new_source)
            self.source_list.extend(new_source)

        examples_idx = self.source_list[:batch_size]
        del self.source_list[:batch_size]

        return self.images[examples_idx,...], self.labels[examples_idx,...]
