import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(hello))
