import numpy
import tensorflow as tf

def is_empty(images, labels):
    empty_idx = []

    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]
        if not image.any() or not label.any():
            empty_idx.append(i)

    print (empty_idx)
    return empty_idx

def image_augmentation(image, label):

    rotation_radian = 3.14
    brightness_delta = 0
    contrast_lower = 0.8
    contrast_upper = 1.2

    image_aug = image
    label_aug = label

    label_aug = tf.cast(label_aug,tf.float32)
    label_aug = label_aug[...,None]

    mix_aug = tf.concat([image_aug,label_aug], axis=3)
    mix_aug = tf.map_fn(tf.image.random_flip_left_right, mix_aug)
    mix_aug = tf.map_fn(tf.image.random_flip_up_down, mix_aug)
    mix_aug = tf.contrib.image.rotate(mix_aug,tf.random_uniform([tf.shape(image)[0]],-rotation_radian,rotation_radian))

    image_aug, label_aug = tf.split(mix_aug,[1,1],3)
    image_aug = tf.image.random_brightness(image_aug, max_delta=brightness_delta)
    image_aug = tf.image.random_contrast(image_aug, lower=contrast_lower, upper=contrast_upper)

    label_aug = tf.cast(label_aug,tf.int64)
    label_aug = tf.squeeze(label_aug)

    return image_aug, label_aug
