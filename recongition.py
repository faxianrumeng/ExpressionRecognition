import tensorflow as tf
import numpy as np
import decodetfrecord as df
from tensorflow.python.framework import graph_util

def bn_parmeter(input):
    scale = tf.Variable(tf.ones([input.get_shape()[-1]]), dtype=tf.float32)
    offset = tf.Variable(tf.zeros([input.get_shape()[-1]]), dtype=tf.float32)
    batch_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), dtype=tf.float32)
    batch_var = tf.Variable(tf.zeros([input.get_shape()[-1]]), dtype=tf.float32)

    batch_mean, batch_var =tf.nn.moments(x=input, axes=[0,1,2])
    return scale, offset, batch_mean, batch_var

def conv_network(height, width, channel):
    conv1_feature = 64
    conv2_feature = 64
    conv3_feature = 64
    max_pool1_size = 2
    max_pool2_size = 2
    max_pool3_size = 2
    full_connect_feature = 256
    keep_prob = 0.99
    x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='inputdata')
    y = tf.placeholder(tf.uint8, shape=[None, 6], name='label')
    step = tf.placeholder(tf.float32, name='coefficient')
    # 权重系数初始化函数
    def weight_vaiable(shape, name='weight'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name, dtype=tf.float32)

    # 初始化偏置
    def bias_variable(shape, name='biases'):
        initial = tf.zeros(shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    # 卷积层函数
    def conv2d(input, kernel):
        return tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')

    # 池化函数
    def pool_max(input, max_pool_size):
        return tf.nn.max_pool(input,
                              ksize=[1, max_pool_size, max_pool_size, 1],
                              strides=[1, max_pool_size, max_pool_size, 1],
                              padding='SAME',
                              name='pool'
                              )
    # 全连接层函数
    def fc(input, w, b):
        return tf.matmul(input, w) + b

    with tf.name_scope('conv1') as scope:
        kernel1 = weight_vaiable(shape=[5, 5, channel, conv1_feature])
        bias1 = bias_variable(shape=[conv1_feature])
        conv1 = tf.nn.bias_add(conv2d(input=x, kernel=kernel1), bias1)
        scale, offset, batch_mean, batch_var = bn_parmeter(conv1)
        bn_conv1 = tf.nn.batch_normalization(conv1,
                                                 mean=batch_mean,
                                                 variance=batch_var,
                                                 offset=offset,
                                                 scale= scale,
                                                 variance_epsilon=0.0001
                                                 )
        pool1 = pool_max(bn_conv1, max_pool1_size)

    with tf.name_scope('conv2') as scope:
        kernel2 = weight_vaiable(shape=[5, 5, conv1_feature, conv2_feature])
        bias2 = bias_variable(shape=[conv2_feature])
        conv2 = tf.nn.bias_add(conv2d(input=pool1, kernel=kernel2), bias2)
        scale, offset, batch_mean, batch_var = bn_parmeter(conv2)
        bn_conv2 = tf.nn.batch_normalization(conv2,
                                                 mean=batch_mean,
                                                 variance=batch_var,
                                                 offset=offset,
                                                 scale=scale,
                                                 variance_epsilon=0.0001
                                                 )
        pool2 = pool_max(bn_conv2, max_pool2_size)

    with tf.name_scope('conv3') as scope:
        kernel3 = weight_vaiable(shape=[3, 3, conv2_feature, conv3_feature])
        bias3 = bias_variable(shape=[conv3_feature])
        conv3 = tf.nn.bias_add(conv2d(input=pool2, kernel=kernel3), bias3)
        scale, offset, batch_mean, batch_var = bn_parmeter(conv3)
        bn_conv3 = tf.nn.batch_normalization(conv3,
                                             mean=batch_mean,
                                             variance=batch_var,
                                             offset=offset,
                                             scale=scale,
                                             variance_epsilon=0.0001
                                             )
        pool3 = pool_max(bn_conv3, max_pool3_size)
        final_conv_shape = pool3.shape.as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_conv_out =tf.reshape(pool3, shape=[-1, final_shape])

    with tf.name_scope('full_connection') as scope:
        weight_fc_1 = weight_vaiable(shape=[final_shape, full_connect_feature])
        bias_fc1 = weight_vaiable(shape=[full_connect_feature])
        fc1 = tf.nn.relu(fc(flat_conv_out, weight_fc_1, bias_fc1))
        drop_out_1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
        weight_fc_2 = weight_vaiable(shape=[full_connect_feature, 6])
        bias_fc2 = weight_vaiable(shape=[6])
        fc2 = tf.nn.relu(fc(drop_out_1, weight_fc_2, bias_fc2))
        net_output = tf.nn.dropout(fc2, keep_prob=keep_prob)

    soft_max_output = tf.nn.softmax(net_output, name='output')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net_output))
    train_step = tf.train.AdamOptimizer(step).minimize(cost)
    prediction_labels = tf.argmax(soft_max_output, axis=1, name='prediction')
    read_label = tf.argmax(y, axis=1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(prediction_labels, read_label), dtype=tf.float32))/192

    return dict(x = x,
                 y = y,
                 cost = cost,
                 accuracy = accuracy,
                 keep_prob = keep_prob,
                 train_step = train_step,
                 step = step)

def net_work(graph, width, height, channel, batch_size, num_epochs, pb_file, root='faceset.tfrecords'):
    batch_set, batch_label = df.single_read_decode(root, batch_size, height, width)
    v_set, v_label = df.single_read_decode(root, 192, height, width, process=1)
    init_step = 0.0001
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num_epochs):
            train_set, train_label = sess.run([batch_set, batch_label])
            sess.run(graph['train_step'], feed_dict={graph['x']: train_set,
                                                     graph['y']: train_label,
                                graph['step']: init_step * pow(0.9, (i/100))})
            val_set, val_label = sess.run([v_set, v_label])
            acc_temp = sess.run(graph['accuracy'], feed_dict={graph['x']: val_set,
                                                              graph['y']: val_label})
            print(u'准确率：', acc_temp)
        coord.request_stop()
        coord.join()
        output_graph_def = graph_util.convert_variables_to_constants(
                                    sess, tf.get_default_graph().as_graph_def(), ['prediction'])
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    coord.request_stop()
    coord.join(threads)
    sess.close()

def main():
    Height = 120
    Width = 120
    Channel = 3
    batch_size = 128
    graph_cov = conv_network(height=Height, width=Width, channel=Channel)
    net_work(graph=graph_cov,
             width=Width,
             height=Height,
             batch_size=batch_size,
             num_epochs=2000,
             channel=3,
             pb_file='conv_net.pb')

main()


