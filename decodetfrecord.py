'''读取tfrecords文件并预处理'''
import tensorflow as tf
import processfile as pf
#多个tfrecord文件时读取
def read_decode(tfrecord_file, batch_size, height, width):
    min_after_dequeue = 1920
    capacity = min_after_dequeue + 3*batch_size
    filename_queue = tf.train.string_input_producer(tfrecord_file)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized,
              features={
              'image_raw': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.string)
              })

    image = features['image_raw']
    decode_image = tf.decode_raw(image, tf.uint8)
    decode_image = tf.reshape(decode_image, shape=[height, width, 3])
    decode_image = tf.to_float(decode_image) * (1.0 / 255)
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [6])
    image_batch, label_batch = tf.train.shuffle_batch(
                               [decode_image, label],
                                  batch_size=batch_size,
                                       capacity=capacity,
                    min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch
#读取一个tfrecord文件
def single_read_decode(tfrecord_file, batch_size, height, width, process=None):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    filename_queue = tf.train.string_input_producer([tfrecord_file])
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
    image = features['image_raw']
    decode_image = tf.decode_raw(image, tf.uint8)
    decode_image = tf.reshape(decode_image, shape=[height, width, 3])
    decode_image = tf.to_float(decode_image) * (1.0/255)
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [6])
    if process is None:
       distort_image = pf.prerocess_image(decode_image, height=120, width=120)
       image_batch, label_batch = tf.train.shuffle_batch(
                                                        [distort_image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    else:
        image_batch, label_batch = tf.train.shuffle_batch(
                                                        [decode_image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

