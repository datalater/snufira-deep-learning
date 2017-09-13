
# coding: utf-8

# In[26]:

import tensorflow as tf
from reader__jmc import reader
from model import dnn

data_reader = reader()
model = dnn()

with tf.Session() as sess:
    
    batch_size = 30
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    max_iter = 1000
    
    for i in range(max_iter):
        x_train, y_train, id_train = data_reader.next_batch(batch_size, split="train")
        
        feed = {model.x : x_train, model.y : y_train}
        
        _, loss_val = sess.run([model.train_op, model.loss], feed_dict=feed)
        
        print ("step: {} | loss_val: {}").format(i, loss_val)
    

# In[ ]:



