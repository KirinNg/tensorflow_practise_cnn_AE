import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True) #28*28*1

def conv(input_data, input_size, output_size):
    w = tf.Variable(tf.random_normal([3,3,input_size,output_size],stddev = 0.1))
    b = tf.Variable(tf.constant(0.1, shape = [output_size]))
    return tf.nn.relu(tf.nn.conv2d(input_data, w, [1,1,1,1], padding = 'SAME') + b)

def pooling(input_data):
    return tf.nn.avg_pool(input_data, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def addLayer(input_data, input_size, output_size):
    W = tf.Variable(tf.random_normal([input_size,output_size],stddev = 0.1))
    basis = tf.Variable(tf.constant(0.1,shape = [output_size]))
    return tf.matmul(input_data, W) + basis

def encoder(input_data):
    waiting_encoder = tf.reshape(input_data, [-1,28,28,1])
    conv_1 = conv(waiting_encoder,1,4) #28*28*1
    pooling_1 = pooling(conv_1) #14*14*4
    conv_2 = conv(pooling_1,4,16) #14*14*16
    pooling_2 = pooling(conv_2) #7*7*16
    conv_3 = conv(pooling_2,16,64) #7*7*64
    pooling_3 = pooling(conv_3) #4*4*64
    conv_4 = conv(pooling_3,64,256) #4*4*256
    pooling_4 = tf.nn.avg_pool(conv_4, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME') #1*1*256
    return pooling_4

def decoder(input_data):
    waiting_decoder = tf.reshape(input_data, [-1,256])
    L_1 = addLayer(waiting_decoder, 256, 784)
    return tf.reshape(L_1, [-1,28,28,1])

X_train = tf.placeholder(tf.float32,[None,784])
X_encoder = encoder(X_train)
X_decoder = decoder(X_encoder)
print(X_decoder._shape)
loss = tf.reduce_sum(tf.pow(tf.reshape(X_train,[-1,28,28,1])-X_decoder,2))/(28*28)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Begin_Train:")
for i in range(40000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={X_train:batch[0]})
    if i%400 == 0:
        print(i/100)
        print(sess.run(loss,feed_dict={X_train:batch[0]}))
print("进行解码测试")
y_pred = X_decoder
encode_decode = sess.run(y_pred, feed_dict={X_train: mnist.test.images[:20]})
fig,a = plt.subplots(2, 20, figsize=(20, 2))
for i in range(20):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()