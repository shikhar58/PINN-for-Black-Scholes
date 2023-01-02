# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 08:45:35 2022

@author: shikhar
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import keras as K
import tensorflow.python.keras.backend as K
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
#tf.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

T=1
K=40
r=0.1
vol=0.4

smin=0
smax=100


"""
delt=0.001
dels=2

smin=0
smax=100

ssteps=int((smax-smin)/dels)

tsteps=int(T/delt)

v=np.zeros((int(tsteps+1),int(ssteps+1)))

s=np.array([smin+dels*i for i in range(ssteps+1)])

v[0,:]=[max(smin+dels*i-K,0) for i in range(ssteps+1)]
v[:,0]=0
v[:,-1]=smax-K
"""
x=np.linspace(0, 100, num=101)
x=x[:,None]
t=np.linspace(0, 0.25, num=250)
t=t[:,None]

ic = np.concatenate((x, 0*x), 1)
x_ic=ic[:,0:1]
t_ic=ic[:,1:2]

boundary=(0,100)
cond_lb = np.concatenate((0*t + boundary[0], t), 1)
x_lb=cond_lb[:,0:1]
t_lb=cond_lb[:,1:2]

cond_rb = np.concatenate((0*t + boundary[1], t), 1)
x_rb=cond_rb[:,0:1]
t_rb=cond_rb[:,1:2]

xx,tt=np.meshgrid(x,t)
xx_f = xx.flatten()[:,None] # NT x 1
tt_f = tt.flatten()[:,None] # NT x 1

layers_c=[2, 50, 50, 50, 50, 50, 1]


def initialize_NN(layers):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.compat.v1.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

#chamka..sigmoid function canged everything

def neural_net( X, weights, biases):
    num_layers = len(weights) + 1
    H = (X-minval)/(maxval-minval)
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        #H = tf.tanh(tf.add(tf.matmul(H, W), b))
        H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


"""
def net_NS_IC( x, y, t):
    c_ic = neural_net(tf.concat([x,y,t], 1), weights, biases)
    return c_ic
  """  
def net_NS( x, t,c_out):
    cx = tf.gradients(c_out, x)[0]
    ct = tf.gradients(c_out, t)[0]
    cxx = tf.gradients(cx, x)[0]
    print("count",1)
    #breakpoint()
    fc=ct-r*x*cx-0.5*vol*x*x*cxx+r*c_out

    return fc

#rho_b*s is state variable as s

def callback(loss):
    print('Loss: %.3e' % (loss))
#X = np.concatenate([x_train, y_train, t_train], 1)



import numpy as np

T=1
K=40
r=0.1
vol=0.4

delt=0.001
dels=2

smin=0
smax=100

ssteps=int((smax-smin)/dels)

tsteps=int(T/delt)

v=np.zeros((int(tsteps+1),int(ssteps+1)))

s=np.array([smin+dels*i for i in range(ssteps+1)])

v[0,:]=[max(smin+dels*i-K,0) for i in range(ssteps+1)]
v[:,0]=0
v[:,-1]=smax-K
for it in range(0,tsteps):
    for ij in range(1,ssteps):
        v[it+1,ij]=delt*v[it,ij+1]*(r*s[ij]/(2*dels)+((vol*s[ij])*(vol*s[ij]))/(2*dels*dels)) + \
                   delt*v[it,ij]*(-(vol*s[ij]/dels)*(vol*s[ij]/dels)-r+1/delt) + \
                   delt*v[it,ij-1]*((vol*s[ij]/dels)*(vol*s[ij]/dels)*0.5-r*s[ij]/(2*dels))
      
final=np.array([v[-i-1,:] for i in range(tsteps+1)])
import matplotlib.pyplot as plt
plt.plot(s[:],final[0,:])
#plt.plot(s[:],v[-1,:])

"""
plt.plot(s[:],v[0,:])
plt.plot(s[:],v[10,:])
plt.plot(s[:],v[20,:])
"""
plt.show()

S=10
E=K
t=0
import math
from math import *




def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0



def analytic(S):
    d1=(log10(S/E)+(r+vol*vol*0.5)*(T-t))/(vol*sqrt(T-t))
    d2=(log10(S/E)+(r-vol*vol*0.5)*(T-t))/(vol*sqrt(T-t))
    
    N1=phi(d1)
    N2=phi(d2)

    first=S*N1
    second=N2*E*e**(-r*(T-t))

    return first-second
    
final_an=np.array([analytic(max(s[i],0.01)) for i in range(len(s))])


#plt.plot(x_ic, catp, 'g', label="actual")
plt.plot(s,final_an)
plt.plot(s[:],final[0,:])
#plt.plot(s[:],final[-1,:])

clean=np.concatenate((s[:,None],np.maximum(0,final_an)[:,None]),1)
import numpy as np 
mu, sigma = 0, 5
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, [np.shape(clean)[0],np.shape(clean)[1]]) 
print(noise)

syn=noise+clean

#plt.plot(x_ic, catp, 'g', label="actual")
plt.plot(s,final_an)
plt.plot(s[:],final[0,:])
plt.scatter(syn[:,0]+15,np.maximum(syn[:,1],0))

x_train1=syn[:,0]+15
y_train1=np.maximum(syn[:,1],0)

syn_data1=np.concatenate((x_train1[:,None],y_train1[:,None]),1)

syn_data=syn_data1[:-7,:]
x_s=syn_data[:,0][:,None]
c_s=syn_data[:,1][:,None]
t_s=np.zeros(len(x_s))[:,None]

#ek baar fir se

#DEFINE THIS ONLY FOR INPUT DATA, IE FEATURES AND TAGER WHICH Has to be minimised
x_i=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t_i=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t_dcb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t_neb=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_f=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
t_f=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

x_ss=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
t_ss=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])



weights_c, biases_c = initialize_NN(layers_c)  

w_ic=tf.Variable(1.0)
w_dc=tf.Variable(10.0)
w_fc=tf.Variable(300.0)
w_nb=tf.Variable(1.0)
w_s=tf.Variable(0*5.0)

#w_fc is 10000 but considering nomralization of loss it will be 10000*(15535/3000)=50000multiplied. then it gives best result



sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

minval=np.array([0,0])
maxval=np.array([100,0.25])
tf_dict = {x_dcb: x_lb, t_dcb: t_lb, x_neb: x_rb, t_neb: t_rb, x_i: x_ic, t_i: t_ic, x_f: xx_f, t_f: tt_f,x_ss: x_s, t_ss: t_s }

#tf_dict = { x_i: x_ic, y_i: y_ic, x_f: x_fp, y_f:y_fp, t_f:t_fp}


#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_ib, biases_ib,np.array([0,1.5]),np.array([0,2.5]))
#c_dcb=neural_net(tf.concat([x_dcb, y_dcb], 1), weights_dcb, biases_dcb,np.array([0,1.5]),np.array([0,2.5]))
c_dcb=neural_net(tf.concat([x_dcb, t_dcb], 1), weights_c, biases_c)
c_neb=neural_net(tf.concat([x_neb, t_neb], 1), weights_c, biases_c)
c_ic=neural_net(tf.concat([x_i, t_i], 1), weights_c, biases_c)
c_f=neural_net(tf.concat([x_f, t_f], 1), weights_c,biases_c)
c_sp=neural_net(tf.concat([x_ss, t_ss], 1), weights_c,biases_c)

fc=net_NS(x_f,t_f,c_f)


c_dc=0
c_nb=smax-K
c_ic1 = np.array([max(*x_ic[i]-K,0) for i in range(len(x_ic))])
c_ic1=c_ic1[:,None]
#loss = 36*tf.reduce_sum(abs(c_ic))+tf.reduce_sum(abs(c_dcb-c_dc))+tf.reduce_sum(abs(f))+tf.reduce_sum(abs(j))
#tf.reduce_sum(abs(f)).eval(feed_dict=tf_dict,session=sess)
#add square 
loss = w_ic*tf.reduce_sum(tf.square(c_ic-c_ic1))/len(x_ic)+w_dc*tf.reduce_sum(tf.square(c_dcb-c_dc))/len(x_lb)+ \
w_nb*tf.reduce_sum(tf.square(c_neb-c_nb))/len(x_rb)+w_fc*tf.reduce_sum(tf.square(fc))/len(xx_f) + \
w_s*tf.reduce_sum(tf.square(c_sp-c_s))/len(x_s)



loss_max=500000-loss



optimizer_Adam = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')

optimizer_Adam_max = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam')


train_op_Adam = optimizer_Adam.minimize(loss,var_list=[weights_c,biases_c])  


train_op_Adam_max = optimizer_Adam_max.minimize(loss_max,var_list=[w_ic,w_dc,w_fc,w_nb])


sess=tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)   
"""
w_icb=[]
w_isb=[]
w_dcb=[]
w_fcb=[]
w_fsb=[]
w_jb=[]
"""
losstot=[]
nIter=20000
for it in range(nIter):
    sess.run(train_op_Adam, tf_dict)
    print(it)
    #fv=f.eval(feed_dict=tf_dict,session=sess)
    #print(it,loss_value,tf.reduce_sum(tf.square(c_dcb-c_dc)).eval(feed_dict=tf_dict,session=sess),tf.reduce_sum(tf.square(f)).eval(feed_dict=tf_dict,session=sess))
    loss_value=loss.eval(feed_dict=tf_dict,session=sess)
    if it%10==0:
        losstot.append(loss_value)

    if loss_value>1:
        if it%1000==0:
            sess.run(train_op_Adam_max, tf_dict)
    
    print(it,loss_value)
    if abs(loss_value)<0.05:
        break


x1=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])
t1=tf.compat.v1.placeholder(tf.float32, shape=[None, x_ic.shape[1]])

t1np = np.empty((101,1))
t1np.fill(0.25)
tf1 = {x1: x_ic, t1: t1np}
cat=neural_net(tf.concat([x1, t1], 1), weights_c, biases_c)
catp=cat.eval(feed_dict=tf1,session=sess)

plt.plot(x_ic, catp, 'g', label="actual")
plt.plot(s,final_an)
plt.plot(s[:],final[0,:])







np.savetxt('pinnoriginal.csv', catp)

"""
loss = w_ic*tf.reduce_sum(tf.square(c_ic-c_ic1))/len(x_ic)+w_dc*tf.reduce_sum(tf.square(c_dcb-c_dc))/len(x_lb)+w_nb*tf.reduce_sum(tf.square(c_neb-c_nb))/len(x_rb)+w_fc*tf.reduce_sum(tf.square(fc))/len(xx_f)

f=w_ic*tf.reduce_sum(tf.square(c_ic-c_ic1))/len(x_ic)
fnp=f.eval(feed_dict=tf_dict,session=sess)

f1=w_dc*tf.reduce_sum(tf.square(c_dcb-c_dc))/len(x_lb)
fnp1=f1.eval(feed_dict=tf_dict,session=sess)

f2=w_nb*tf.reduce_sum(tf.square(c_neb-c_nb))/len(x_rb)
fnp2=f2.eval(feed_dict=tf_dict,session=sess)



f3=w_fc*tf.reduce_sum(tf.square(fc))/len(xx_f)
fnp3=f3.eval(feed_dict=tf_dict,session=sess)"""