# -*- coding: utf-8 -*-
import pdb
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
#inputdata
filename1='human.txt'
filename2='animal.txt'
filename3='human_labels.txt'
def getdata(filename):
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
            if spindle==100:
                spindle=0
                data.append(data50)
                data50=[]
    return(data)
def getlabels(filename):
    i=0
    data=[]
    with open(filename) as f:
        for line in f :
            numbers_str = line.split()
            numbers_float = [float(x) for x in numbers_str]
            data.append(numbers_float[0])
    return(data)


#random variables
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
#discriminator
X = tf.placeholder(tf.float32, shape=[None, 256])

D_W1= tf.Variable(xavier_init([4940, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_filter1= tf.Variable(xavier_init([10,1,20]))
D_conb1 = tf.Variable(tf.zeros(shape=[20]))

D_W2= tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))


theta_D = [D_W1, D_W1, D_filter1,D_conb1,D_W2,D_b2,D_W3,D_b3,D_W4,D_b4]

#extractor
Z = tf.placeholder(tf.float32, shape=[None, 100])

E_W1= tf.Variable(xavier_init([3640,2048 ]))
E_b1 = tf.Variable(tf.zeros(shape=[2048]))

E_filter1= tf.Variable(xavier_init([10,1,40]))
E_conb1 = tf.Variable(tf.zeros(shape=[40]))

E_W2= tf.Variable(xavier_init([2048, 1024]))
E_b2 = tf.Variable(tf.zeros(shape=[1024]))

E_W3= tf.Variable(xavier_init([1024, 512]))
E_b3 = tf.Variable(tf.zeros(shape=[512]))

E_W4= tf.Variable(xavier_init([256, 256]))
E_b4 = tf.Variable(tf.zeros(shape=[400]))

R_W1= tf.Variable(xavier_init([256,32 ]))
R_b1 = tf.Variable(tf.zeros(shape=[32]))

R_W2= tf.Variable(xavier_init([32, 1]))
R_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_E = [E_W1, E_b1, E_filter1,E_conb1,E_W2,E_b2,E_W3,E_b3,E_W4,E_b4,R_W1, R_b1,R_W2,R_b2]

#
def extractor(z):
    x=tf.expand_dims(z,-1)
    x1=tf.nn.conv1d(x, E_filter1, 1, 'VALID')+E_conb1
    y=tf.reshape(x1,[-1,3640])
    y1=tf.tanh(tf.matmul(y, E_W1) + E_b1)
    y2=tf.tanh(tf.matmul(y1, E_W2) + E_b2)
    y3=tf.tanh(tf.matmul(y2, E_W3) + E_b3)
    E_prob=tf.matmul(y3, E_W4) + E_b4
    return E_prob
#
def result(x):
    y1=tf.tanh(tf.matmul(y, R_W1) + R_b1)
    R=tf.matmul(y1, R_W2) + R_b2   

    return(R)
def discriminator(x):
    x1=tf.expand_dims(x,-1)
    x2=tf.nn.conv1d(x1, D_filter1, 1, 'VALID')+D_conb1
    y=tf.reshape(x2,[-1,4940])
    y1=tf.tanh(tf.matmul(y, D_W1) + D_b1)
    D_logit=tf.matmul(y1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob,D_logit

#losses
human= extractor(Z1)
animal=extractor(Z2)
result = result(human)


D_human, D_logit_human= discriminator(human)
D_animal, D_logit_animal = discriminator(human)


D_loss_human = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_human, labels=tf.ones_like(D_logit_human)))
D_loss_animal= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_animal, labels=tf.zeros_like(D_logit_animal)))
D_loss = D_loss_human+ D_loss_animal

E_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_animal, labels=tf.ones_like(D_logit_animal)))
E_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result, labels=labels))

a=0.5

E_loss=E_loss1+a*E_loss2


D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=theta_E)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


humandata=getdata(filename1)
animaldata=getdata(filename2)
humanlabels=getlabels(filename3)
for it in range(200000):
    H_mb= humandata[(it % 860)*25:(it % 860)*25+25]
    A_mb= animaldata[(it % 860)*25:(it % 860)*25+25]
    L_mb= humanlabels[(it % 860)*25:(it % 860)*25+25]
    #pdb.set_trace()
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={Z1:H_mb, Z2:A_mb })

    _, E_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z1: H_mb,Z2:A_mb ,labels:L_mb })
    if it % 1000 == 0:
        print('Iter: {}'.format(it))

        print('D loss: {:.4}'. format(D_loss_curr))

        print('E_loss: {:.4}'.format(E_loss_curr))


