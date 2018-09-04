# -*- coding: utf-8 -*-
import pdb
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
#读入数据
filename='new_spindles.txt'
def getdata():
    i=0
    spindle=0
    data=[]
    data50=[]
    with open(filename) as f:
        for line in f :
            i=i+1
            spindle=spindle+1
            numbers_str = line.split()
            numbers_float = [float(x) for x in numbers_str]
            data50.append(numbers_float[0])
            if spindle==200:
                spindle=0
                data.append(data50)
                data50=[]
    return(data)

#从正态分布输出随机值
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
#判别模型的输入和参数初始化
X = tf.placeholder(tf.float32, shape=[None, 200])

D_W1= tf.Variable(xavier_init([7120, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

D_filter1= tf.Variable(xavier_init([20,1,40]))
D_conb1 = tf.Variable(tf.zeros(shape=[40]))

D_filter2= tf.Variable(xavier_init([4,40,40]))
D_conb2 = tf.Variable(tf.zeros(shape=[40]))

theta_D = [D_W1, D_W2, D_b1, D_b2, D_filter1,D_conb1,D_filter2,D_conb2]

#生成模型的输入和参数初始化
Z = tf.placeholder(tf.float32, shape=[None, 100])

G_filter1=tf.Variable(xavier_init([20,1,20]))
G_conb1=tf.Variable(tf.zeros(shape=[1]))

G_filter2=tf.Variable(xavier_init([4,20,20]))
G_conb2=tf.Variable(tf.zeros(shape=[20]))

G_W1 = tf.Variable(xavier_init([100,512]))
G_b1 = tf.Variable(tf.zeros(shape=[512]))

G_W2 = tf.Variable(xavier_init([512, 3560]))
G_b2 = tf.Variable(tf.zeros(shape=[3560]))
G_W3 = tf.Variable(xavier_init([200,200]))
G_b3 = tf.Variable(tf.zeros(shape=[200]))
theta_G = [G_W1, G_W2, G_b1, G_b2,G_filter1,G_conb1,G_filter2,G_conb2,G_W3,G_b3]

#随机噪声采样函数
def sample_Z(m,n):
    return np.random.uniform(-1., 1., size=[m, n])
#生成模型
def generator(z):
    t=tf.shape(z)
    p=t[0]
    G_h1 = tf.matmul(z, G_W1) + G_b1
    G_h2 = tf.matmul(G_h1, G_W2) + G_b2
    x=tf.reshape(G_h2,[-1,178,20])
    x1=tf.contrib.nn.conv1d_transpose(x, G_filter2,[p,181,20], 1, 'VALID')+G_conb2
    y=tf.sinh(tf.contrib.nn.conv1d_transpose(x1, G_filter1,[p,200,1], 1, 'VALID')+G_conb1)
    G_h3=tf.reshape(y,[p,200])
    G_h5= tf.matmul(G_h3, G_W3) + G_b3
    G_prob=G_h3
    return G_prob
#判别模型
def discriminator(x):
    x=tf.expand_dims(x,-1)
    x1=tf.nn.conv1d(x, D_filter1, 1, 'VALID')+D_conb1
    y=tf.nn.conv1d(x1, D_filter2, 1, 'VALID')+D_conb2
    y1=tf.reshape(y,[-1,7120])
    #pdb.set_trace()
    D_h1 =tf.nn.tanh(tf.matmul(y1, D_W1) + D_b1)

    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.tanh(D_logit)
    return D_prob, D_logit
#画图函数

#喂入数据
G_sample= generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# 计算losses:
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 25
Z_dim = 100



sess = tf.Session()
sess.run(tf.global_variables_initializer())
if not os.path.exists('out/'):
    os.makedirs('out/')
i = 0
data=getdata()
for it in range(300000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
        x=np.reshape(np.arange(200),[200,1])
        y=np.reshape(np.asarray(samples),[200,1])
        plt.figure()
        plt.plot(x,y)
        i=i+1
        print(i)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close
    X_mb= data[(it % 7500)*25:(it % 7500)*25+25]
    #pdb.set_trace()
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    if it % 1000 == 0:
        print('Iter: {}'.format(it))

        print('D loss: {:.4}'. format(D_loss_curr))

        print('G_loss: {:.4}'.format(G_loss_curr))

generating=[]
for kk in range(15):
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
    y=np.reshape(np.asarray(samples),[200,1])
    generating.append(y)
outfile=[]
with open('generated_spindle.txt','w') as f1:
    for k in range(15):
        random_number=150
        zero0=np.zeros(random_number)
        for kkk in range(200):
            nnn=float(generating[k][kkk])
            outfile.append(nnn)
        for kkk in range(random_number):
            outfile.append(0)
    f1.write(str(outfile))
