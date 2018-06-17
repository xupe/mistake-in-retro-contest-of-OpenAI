import numpy as np
import tensorflow as tf
import math

w_initializer = tf.random_normal_initializer(0., 0.3)
b_initializer = tf.constant_initializer(0.1)

def nature_cnn(obs_batch, dense=tf.layers.dense):
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.orthogonal_initializer(gain=math.sqrt(2)),
    }
    with tf.variable_scope('layer_1'):
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 8, 4, **conv_kwargs)
        print(cnn_1)
    with tf.variable_scope('layer_2'):
        cnn_2 = tf.layers.conv2d(cnn_1, 64, 4, 2, **conv_kwargs)
    with tf.variable_scope('layer_3'):
        cnn_3 = tf.layers.conv2d(cnn_2, 64, 3, 1, **conv_kwargs)
    flat_size = product([x.value for x in cnn_3.get_shape()[1:]])
    flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))
    #weight_noisy, bias_noisy = sample_noise(flat_size, 512)
    return dense(flat_in, 512, **conv_kwargs)

def product(prob):
    p = 1
    for pro in prob:
        p *= pro
    return p

def noisy_net_dense(inputs, units, activation=None, sigma0=0.5,
                    #weight_noisy=None, bias_noisy=None,
                    kernel_initializer=None, name=None, reuse=None):
    num_inputs = inputs.get_shape()[-1].value
    stddev = 1 / math.sqrt(num_inputs)
    noise_stddev = sigma0 / math.sqrt(num_inputs)
    if activation is None:
        activation = lambda x : x
    if kernel_initializer is None:
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)

    with tf.variable_scope(None, default_name=(name or 'noisy_layer'), reuse=reuse):
        weight_mean = tf.get_variable('weight_mu', shape=(num_inputs, units),
                                      initializer=kernel_initializer)
        bias_mean = tf.get_variable('bias_mu', shape=(1, units),
                                    initializer=kernel_initializer)
        stddev *= sigma0
        weight_stddev = tf.get_variable('weight_sigma', shape=(num_inputs, units),
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_stddev = tf.get_variable('bias_sigma', shape=(units,),
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        weight_noisy, bias_noisy = sample_noise(num_inputs, units, type=bias_stddev.dtype.base_dtype, stddev=noise_stddev)
        #weight_noisy = tf.get_variable('noisy_weight', weight_noisy_tmp)
        #bias_noisy = tf.get_variable('noisy_bias', bias_noisy_tmp)
        return activation(tf.matmul(inputs, weight_mean + weight_stddev * weight_noisy)
                          + bias_mean + bias_stddev * bias_noisy)

def sample_noise(inputs_num, units, stddev=1, type=tf.float32):
    bias_noisy = tf.random_normal((units,), dtype=type, stddev=stddev)
    weight_noisy = _factorized(inputs_num, units, stddev=stddev)
    print('generate the noisy')
    print(weight_noisy)
    print(bias_noisy)
    return weight_noisy, bias_noisy

def _factorized(inputs, outputs, stddev=1):
    noise1 = _signed_sqrt(tf.random_normal((inputs, 1), stddev=stddev))
    noise2 = _signed_sqrt(tf.random_normal((1, outputs), stddev=stddev))
    return tf.matmul(noise1, noise2)

def _signed_sqrt(values):
    return tf.sqrt(tf.abs(values)) * tf.sign(values)


def nature_cnn_add_one_layer(obs_batch, dense=tf.layers.dense):
    #start 80 * 80 * 3
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.contrib.layers.xavier_initializer(),
        'padding': 'SAME'
    }
    with tf.variable_scope('layer_1'):
        #cnn_1 = CNN_layer(obs_batch, 32, 8, 2, True)
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 8, 2, **conv_kwargs)
    with tf.variable_scope('layer_2'):
        #cnn_2 = CNN_layer(cnn_1, 64, 4, 2, True)
        cnn_2 = tf.layers.conv2d(cnn_1, 64, 4, 2, **conv_kwargs)
    with tf.variable_scope('layer_3'):
        #cnn_3 = CNN_layer(cnn_2, 128, 3, 2, True)
        cnn_3 = tf.layers.conv2d(cnn_2, 128, 3, 2, **conv_kwargs)
    with tf.variable_scope('layer_4'):
        #cnn_4 = CNN_layer(cnn_3, 256, 3, 2, True)
        cnn_4 = tf.layers.conv2d(cnn_3, 128, 3, 1, **conv_kwargs)
    flat_size = product([x.value for x in cnn_4.get_shape()[1:]])
    flat_in = tf.reshape(cnn_4, (tf.shape(cnn_4)[0], int(flat_size)))
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.contrib.layers.xavier_initializer(),
        }
    return dense(flat_in, 512, **conv_kwargs)

def CNN_layer(input, filters, kernel_size, strides, use_bias=False):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=use_bias, name='conv')
    bn = _bn(conv, True, name='bn')
    relu = tf.nn.relu(bn, name='relu')
    return relu

def my_net(obs_batch, dense=tf.layers.dense):
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.contrib.layers.xavier_initializer(),
    }
    with tf.variable_scope('layer_1'):
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 8, 4, **conv_kwargs)
        print(cnn_1)
    with tf.variable_scope('layer_2'):
        with tf.variable_scope('first'):
            conv1 = tf.layers.conv2d(cnn_1, filters=64, kernel_size=3, strides=2, padding='SAME',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             use_bias=False, name='conv1')
            bn1 = _bn(conv1, True, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')

            conv2 = tf.layers.conv2d(relu1, filters=64, kernel_size=3, strides=1, padding='SAME',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=False, name='conv2')
            bn2 = _bn(conv2, True, name='bn2')

            conv3 = tf.layers.conv2d(cnn_1, filters=64, kernel_size=1, strides=2, padding='SAME',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    use_bias=False, name='conv3')
            bn3 = _bn(conv3, True, name='bn3')

            relu_first = tf.nn.relu(bn2 + bn3, name='first_relu')

        with tf.variable_scope('second'):
            conv1 = tf.layers.conv2d(relu_first, filters=64, kernel_size=3, strides=1, padding='SAME',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=False, name='conv1')
            bn1 = _bn(conv1, True, name='bn1')
            relu1 = tf.nn.relu(bn1, name='relu1')
            conv2 = tf.layers.conv2d(relu1, filters=64, kernel_size=3, strides=1, padding='SAME',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     use_bias=False, name='conv2')
            bn2 = _bn(conv2, True, name='bn2')
            relu2 = tf.nn.relu(bn2, name='relu2')

    with tf.variable_scope('layer3'):
        cnn_3 = tf.layers.conv2d(relu2, 128, 3, 2, **conv_kwargs)

    flat_size = product([x.value for x in cnn_3.get_shape()[1:]])
    flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))
    #weight_noisy, bias_noisy = sample_noise(flat_size, 512)
    return dense(flat_in, 512, **conv_kwargs)


def _bn(x, is_train, global_step=None, name='bn'):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
        # decay = moving_average_decay
        # else:
        # decay = tf.cond(tf.greater(global_step, 100)
        # , lambda: tf.constant(moving_average_decay, tf.float32)
        # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                                 initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                                   initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
        # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        if is_train:
            mean, var = batch_mean, batch_var
        else:
            mean, var = mu, sigma
            #mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
            #lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
        # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
        # scale=True, epsilon=1e-5, is_training=is_train,
        # trainable=True)
    return bn
