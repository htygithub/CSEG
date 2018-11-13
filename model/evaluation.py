import tensorflow as tf

def loss_calc(logits, labels):
    labels = tf.squeeze(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    losses = tf.reduce_mean(cross_entropy)
    #tf.summary.scalar('loss', loss)

    l2_loss = []
    for v in tf.trainable_variables():
        if 'BatchNorm' not in v.name and 'weights' in v.name:
            l2_loss.append(tf.nn.l2_loss(v))
    #loss = losses + tf.add_n(l2_loss)*0.000001
    loss = losses
    return loss

def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def get_dice(logits, labels):
    labels = tf.squeeze(labels)
    labels = tf.to_int32(labels)
    #logits = tf.to_int64(logits)
    # reshape to match args required for the top_k function

    logits_re = tf.reshape( logits, [-1, tf.shape(logits)[-1]] )
    labels_re = tf.reshape( labels, [-1, 1] )

    logits_re = tf.cast(tf.argmax(logits_re, 1), tf.int32)
    logits_re = tf.reshape( logits_re, [-1, 1] )

    labels_re = tf.cast(tf.greater(labels_re,0), tf.int32)
    indices = tf.cast(tf.greater(logits_re,0), tf.int32)

    example_sum = tf.reduce_sum(tf.cast(indices, tf.int32))
    label_sum = tf.reduce_sum(tf.cast(labels_re, tf.int32))
    sum_tensor = tf.add(indices, tf.cast( labels_re, tf.int32 ))
    twos = tf.fill( sum_tensor.get_shape(), 2 )
    intersection_tensor = tf.div( sum_tensor, twos )
    intersection_sum = tf.reduce_sum( tf.cast( intersection_tensor,
                                           tf.int32 ) )
    intersection_sum = tf.to_float(intersection_sum)
    label_sum = tf.to_float(label_sum)
    example_sum = tf.to_float(example_sum)
    precision = (2.0 * intersection_sum) / ( label_sum + example_sum )
    #print('OUTPUT:  Dice metric = %.3f' %  precision)

    return precision
