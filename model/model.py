# ============================================================== #
#                            U-net                               #
#                                                                #
#                                                                #
# Unet tensorflow implementation                                 #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import numpy as np


def conv(inputs, kernel_size, num_outputs, name,
        stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution layer followed by activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv_btn(inputs, kernel_size, num_outputs, name,
        is_training = True, stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution layer followed by batch normalization then activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        is_training: Boolean, in training mode or not
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, bias)
        outputs = tf.contrib.layers.batch_norm(outputs, center = True, scale = True, is_training = is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def deconv(inputs, kernel_size, num_filters_in, num_outputs, name,
        stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution Transpose followed by activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_filters_in: Ingteger, number of channels in input tensor
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], num_outputs, num_filters_in]
        stride_shape = [1, stride_size[0], stride_size[1], 1]
        input_shape  = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], input_shape[1], input_shape[2], num_outputs])

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv_trans = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        outputs    = tf.nn.bias_add(conv_trans, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def deconv_btn(inputs, kernel_size, num_filters_in, num_outputs, name,
        is_training = True, stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution Transpose followed by batch normalization then activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_filters_in: Ingteger, number of channels in input tensor
        num_outputs: Integer, number of convolution filters
        is_training: Boolean, in training mode or not
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], num_outputs, num_filters_in]
        stride_shape = [1, stride_size[0], stride_size[1], 1]
        input_shape  = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], input_shape[1], input_shape[2], num_outputs])

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv_trans = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        outputs    = tf.nn.bias_add(conv_trans, bias)
        outputs    = tf.contrib.layers.batch_norm(outputs, center = True, scale = True, is_training = is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def deconv_upsample(inputs, factor, name, padding = 'SAME', activation_fn = None):
    """
    Convolution Transpose upsampling layer with bilinear interpolation weights:
    ISSUE: problems with odd scaling factors
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        factor: Integer, upsampling factor
        name: String, scope name
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height * factor, width * factor, num_filters_in]
    """

    with tf.variable_scope(name):
        stride_shape   = [1, factor, factor, 1]
        input_shape    = tf.shape(inputs)
        num_filters_in = inputs.get_shape()[-1].value
        output_shape   = tf.stack([input_shape[0], input_shape[1] * factor, input_shape[2] * factor, num_filters_in])

        weights = bilinear_upsample_weights(factor, num_filters_in)
        outputs = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def bilinear_upsample_weights(factor, num_outputs):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization:
    ----------
    Args:
        factor: Integer, upsampling factor
        num_outputs: Integer, number of convolution filters

    Returns:
        outputs: Tensor, [kernel_size, kernel_size, num_outputs]
    """

    kernel_size = 2 * factor - factor % 2

    weights_kernel = np.zeros((kernel_size,
                               kernel_size,
                               num_outputs,
                               num_outputs), dtype = np.float32)

    rfactor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = rfactor - 1
    else:
        center = rfactor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    upsample_kernel = (1 - abs(og[0] - center) / rfactor) * (1 - abs(og[1] - center) / rfactor)

    for i in range(num_outputs):
        weights_kernel[:, :, i, i] = upsample_kernel

    init = tf.constant_initializer(value = weights_kernel, dtype = tf.float32)
    weights = tf.get_variable('weights', weights_kernel.shape, tf.float32, init)

    return weights


def batch_norm(inputs, name, is_training = True, decay = 0.9997, epsilon = 0.001, activation_fn = None):
    """
    Batch normalization layer (currently using Tf-Slim):
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        name: String, scope name
        is_training: Boolean, in training mode or not
        decay: Float, decay rate
        epsilon, Float, epsilon value
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    return tf.contrib.layers.batch_norm(inputs, name = name, decay = decay,
                            center = True, scale = True,
                            is_training = is_training,
                            epsilon = epsilon, activation_fn = activation_fn)


def flatten(inputs, name):
    """
    Flatten input tensor:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height * width * channels]
    """

    with tf.variable_scope(name):
        dim     = inputs.get_shape()[1:4].num_elements()
        outputs = tf.reshape(inputs, [-1, dim])

        return outputs


def fully_connected(inputs, num_outputs, name, activation_fn = tf.nn.relu):
    """
    Fully connected layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        num_outputs: Integer, number of output neurons
        name: String, scope name
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value

        weights = tf.get_variable('weights', [num_filters_in, num_outputs], tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        outputs = tf.matmul(inputs, weights)
        outputs = tf.nn.bias_add(outputs, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def maxpool(inputs, kernel_size, name, padding = 'SAME'):
    """
    Max pooling layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding

    Returns:
        outputs: Tensor, [batch_size, height / kernelsize[0], width/kernelsize[1], channels]
    """

    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]

    outputs = tf.nn.max_pool(inputs, ksize = kernel_shape,
            strides = kernel_shape, padding = padding, name = name)

    return outputs


def dropout(inputs, keep_prob, name):
    """
    Dropout layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        keep_prob: Float, probability of keeping this layer
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    return tf.nn.dropout(inputs, keep_prob = keep_prob, name = name)


def concat(inputs1, inputs2, name):
    """
    Concatente two tensors:
    ----------
    Args:
        inputs1: Tensor, [batch_size, height, width, channels]
        inputs2: Tensor, [batch_size, height, width, channels]
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height, width, channels1 + channels2]
    """

    return tf.concat(axis=3, values=[inputs1, inputs2], name = name)


def add(inputs1, inputs2, name, activation_fn = None):
    """
    Add two tensors:
    ----------
    Args:
        inputs1: Tensor, [batch_size, height, width, channels]
        inputs2: Tensor, [batch_size, height, width, channels]
        name: String, scope name
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    with tf.variable_scope(name):
        outputs = tf.add(inputs1, inputs2)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs



def inference(color_inputs, is_training):
    num_classes = 4

    """
    Build unet network:
    ----------
    Args:
        color_inputs: Tensor, [batch_size, height, width, 3]
        num_classes: Integer, number of segmentation (annotation) labels
        is_training: Boolean, in training mode or not (for dropout & bn)
    Returns:
        logits: Tensor, predicted annotated image flattened
                              [batch_size * height * width,  num_classes]
    """

    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)

    # Encoder Section
    # Block 1
    color_conv1_1 = conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)
    color_conv1_2 = conv_btn(color_conv1_1, [3, 3], 64, 'conv1_2', is_training = is_training)
    color_pool1   = maxpool(color_conv1_2, [2, 2],  'pool1')

    # Block 2
    color_conv2_1 = conv_btn(color_pool1,   [3, 3], 128, 'conv2_1', is_training = is_training)
    color_conv2_2 = conv_btn(color_conv2_1, [3, 3], 128, 'conv2_2', is_training = is_training)
    color_pool2   = maxpool(color_conv2_2, [2, 2],   'pool2')

    # Block 3
    color_conv3_1 = conv_btn(color_pool2,   [3, 3], 256, 'conv3_1', is_training = is_training)
    color_conv3_2 = conv_btn(color_conv3_1, [3, 3], 256, 'conv3_2', is_training = is_training)
    color_pool3   = maxpool(color_conv3_2, [2, 2],   'pool3')
    color_drop3   = dropout(color_pool3, dropout_keep_prob, 'drop3')

    # Block 4
    color_conv4_1 = conv_btn(color_drop3,   [3, 3], 512, 'conv4_1', is_training = is_training)
    color_conv4_2 = conv_btn(color_conv4_1, [3, 3], 512, 'conv4_2', is_training = is_training)
    color_pool4   = maxpool(color_conv4_2, [2, 2],   'pool4')
    color_drop4   = dropout(color_pool4, dropout_keep_prob, 'drop4')

    # Block 5
    color_conv5_1 = conv_btn(color_drop4,   [3, 3], 1024, 'conv5_1', is_training = is_training)
    color_conv5_2 = conv_btn(color_conv5_1, [3, 3], 1024, 'conv5_2', is_training = is_training)
    color_drop5   = dropout(color_conv5_2, dropout_keep_prob, 'drop5')

    # Decoder Section
    # Block 1
    upsample6     = deconv_upsample(color_drop5, 2,  'upsample6')
    concat6       = concat(upsample6, color_conv4_2, 'contcat6')
    color_conv6_1 = conv_btn(concat6,       [3, 3], 512, 'conv6_1', is_training = is_training)
    color_conv6_2 = conv_btn(color_conv6_1, [3, 3], 512, 'conv6_2', is_training = is_training)
    color_drop6   = dropout(color_conv6_2, dropout_keep_prob, 'drop6')

    # Block 2
    upsample7     = deconv_upsample(color_drop6, 2,  'upsample7')
    concat7       = concat(upsample7, color_conv3_2, 'concat7')
    color_conv7_1 = conv_btn(concat7,       [3, 3], 256, 'conv7_1', is_training = is_training)
    color_conv7_2 = conv_btn(color_conv7_1, [3, 3], 256, 'conv7_2', is_training = is_training)
    color_drop7   = dropout(color_conv7_2, dropout_keep_prob, 'drop7')

    # Block 3
    upsample8     = deconv_upsample(color_drop7, 2,  'upsample8')
    concat8       = concat(upsample8, color_conv2_2, 'concat8')
    color_conv8_1 = conv_btn(concat8,       [3, 3], 128, 'conv8_1', is_training = is_training)
    color_conv8_2 = conv_btn(color_conv8_1, [3, 3], 128, 'conv8_2', is_training = is_training)

    # Block 4
    upsample9     = deconv_upsample(color_conv8_2, 2, 'upsample9')
    concat9       = concat(upsample9, color_conv1_2,  'concat9')
    color_conv9_1 = conv_btn(concat9,       [3, 3], 64,   'conv9_1', is_training = is_training)
    color_conv9_2 = conv_btn(color_conv9_1, [3, 3], 64,   'conv9_2', is_training = is_training)

    # Block 5
    logits  = conv(color_conv9_2, [1, 1], num_classes, 'logits', activation_fn = None)
    #logits = tf.reshape(score, (-1, num_classes))

    return logits


def segmentation_loss(logits, labels, class_weights = None):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size * height * width, num_classes]
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None

    Returns:
        segment_loss: Segmentation loss
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels = labels, logits = logits, name = 'segment_cross_entropy_per_example')

    if class_weights is not None:
        weights = tf.matmul(labels, class_weights, a_is_sparse = True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = tf.multiply(cross_entropy, weights)

    segment_loss  = tf.reduce_mean(cross_entropy, name = 'segment_cross_entropy')

    tf.summary.scalar("loss/segmentation", segment_loss)

    return segment_loss


def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """

    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    tf.summary.scalar("loss/weights", l2_loss)

    return l2_loss


def loss(logits, labels, weight_decay_factor, class_weights = None):
    """
    Total loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]
        weight_decay_factor: float, factor with which weights are decayed
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None

    Returns:
        total_loss: Segmentation + Classification losses + WeightDecayFactor * L2 loss
    """

    segment_loss = segmentation_loss(logits, labels, class_weights)
    total_loss   = segment_loss + weight_decay_factor * l2_loss()

    tf.summary.scalar("loss/total", total_loss)

    return total_loss
