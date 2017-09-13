
# coding: utf-8

# In[4]:

import tensorflow as tf

class dnn(object):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 6])
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        
        self.w1 = tf.get_variable(name='w1', shape=[6, 50])
        self.b1 = tf.get_variable(name='b1', shape=[50])
        
        self.w2 = tf.get_variable(name='w2', shape=[50, 50])
        self.b2 = tf.get_variable(name='b2', shape=[50])
        
        self.w3 = tf.get_variable(name='w3', shape=[50, 2])
        self.b3 = tf.get_variable(name='b3', shape=[2])
        
        self.build_graph()
        
    def build_graph(self):
        
        h1 = tf.matmul(self.x, self.w1) + self.b1
        h1 = tf.nn.relu(h1)
        
        h2 = tf.matmul(h1, self.w2) + self.b2
        h2 = tf.nn.relu(h2)
        
        h3 = tf.matmul(h2, self.w3) + self.b3
        h3 = tf.nn.softmax(h3,dim=-1)
        
        prediction = h3
        
        y_onehot = tf.one_hot(indices=self.y, depth=2, on_value=1.0, off_value=0.0)   # depth = y의 label 종류 
        self.loss = tf.reduce_mean(tf.pow(prediction - y_onehot, 2))
        
        optimzer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.train_op = optimzer.minimize(self.loss)
        
        
        
        
        


# In[ ]:



