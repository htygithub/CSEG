import os
import random
import numpy as np

import tensorflow as tf
import model as md

AUGMENT_IMAGE = False
REMOVE_EMPTY = False
HAVE_GPU = False
SAVE_INTERVAL = 100


mat = 'STACOM_res256X256.mat'

RUN_NAME = "U-net_HS"
CONV_SIZE = 3
MAX_STEPS = 200
BATCH_SIZE = 6
BATCH_NORM_DECAY_RATE = 0.9

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

def main():
    training_images, training_labels, test_images, test_labels = md.Load_Data(mat)

    if REMOVE_EMPTY:
        empty_idx = md.is_empty(training_images, training_labels)
        images_transform = np.delete(training_images, empty_idx, 0)
        labels_transform = np.delete(training_labels, empty_idx, 0)
        training_data = md.GetData(images_transform, labels_transform)
    else:
        training_data = md.GetData(training_images, training_labels)

    test_data = md.GetData(test_images, test_labels)

    g = tf.Graph()
    with g.as_default():

        images = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 1])
        labels = tf.placeholder(tf.int64, [BATCH_SIZE, 256, 256])
        is_training = tf.placeholder(tf.bool)

        if AUGMENT_IMAGE:
            images, labels = md.image_augmentation(images, labels)

        logits = md.inference(images, is_training)

        loss = md.loss_calc(logits=logits, labels=labels)

        train_op, global_step = md.training(loss=loss, learning_rate=1e-04)

        accuracy = md.evaluation(logits=logits, labels=labels)

        dice = md.get_dice(logits=logits, labels=labels)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])

        sm = tf.train.SessionManager()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

            train_writer = tf.summary.FileWriter(LOG_DIR + "/Train", sess.graph)
            test_writer = tf.summary.FileWriter(LOG_DIR + "/Test")

            global_step_value, = sess.run([global_step])

            print("Last trained iteration was: ", global_step_value)

            for step in range(global_step_value+1, global_step_value+MAX_STEPS+1):

                print("Iteration: ", step)

                images_batch, labels_batch = training_data.next_batch(BATCH_SIZE)

                train_feed_dict = {images: images_batch,
                                   labels: labels_batch,
                                   is_training: True}

                train_dice_value, _, train_loss_value, train_accuracy_value, train_summary_str = sess.run([dice, train_op, loss, accuracy, summary], feed_dict=train_feed_dict)

                if step % SAVE_INTERVAL == 0:

                    print("Train Loss: ", train_loss_value)
                    print("Train accuracy: ", train_accuracy_value)
                    print("Train dice: ", train_dice_value)
                    train_writer.add_summary(train_summary_str, step)
                    train_writer.flush()

                    images_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                    test_feed_dict = {images: images_batch,
                                      labels: labels_batch,
                                      is_training: False}

                    test_dice_value, test_loss_value, test_accuracy_value, test_summary_str = sess.run([dice, loss, accuracy, summary], feed_dict=test_feed_dict)

                    print("Test Loss: ", test_loss_value)
                    print("Test accuracy: ", test_accuracy_value)
                    print("Test dice: ", test_dice_value)
                    test_writer.add_summary(test_summary_str, step)
                    test_writer.flush()

                    saver.save(sess, CHECKPOINT_FL, global_step=step)
                    print("Session Saved")
                    print("================")

if __name__ == '__main__':
    main()
