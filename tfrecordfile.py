'''将图像数据转为tfrecords格式'''
import tensorflow as tf
import numpy as np
import os
import cv2

def  _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#将image数据转为TFRecord格式
height = 120
width = 120
channels = 3
filedir = []
dir = 'dfacedata'
for r, d, f in os.walk(dir):
    filedir.append(f)
ori_label = np.zeros([len(filedir[0]), 6], dtype=np.uint8)
image_array = np.zeros([len(filedir[0]), width, height, channels],
                       dtype=np.uint8)
for i in range(len(filedir[0])):
    ori_label[i, i//32] = 1
    image = cv2.imread(dir + '/' + filedir[0][i])
    image = cv2.resize(image, (height, width))
    image_array[i] = np.array(image)

writer = tf.python_io.TFRecordWriter('faceset.tfrecords')
for j in range(len(image_array)):
    label = ori_label[j].tostring()
    image_raw = image_array[j].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
                               'image_raw': _bytes_feature(image_raw),
                               'label': _bytes_feature(label)
                              }))
    writer.write(example.SerializeToString())

writer.close()
