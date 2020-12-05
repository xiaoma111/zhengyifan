# -*- coding:utf-8 -*-
import tensorflow as tf



a = tf.Variable(0, name='a')

with tf.device('/gpu:0'):

    b = tf.Variable(0, name='b')
  # 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU上

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

sess.run(tf.initialize_all_variables())

