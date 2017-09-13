import tensorflow as tf
from reader__jmc import reader
from model import dnn

# Loading data reader at Practice(2)
data_reader = reader()

# Loading DNN graph at Practice(3)
model = dnn()

# Define Session for running graph
# and initialize model's parameters
sess = tf.Session()
sess.run(tf.global_variables_initializer())


batch_size = 16
max_steps = 100000
for i in range(max_steps):
  # For each iteration, first we get batch x&y data
  x_train, y_train, id_train = data_reader.next_batch(16, split="train")

  # Next, construct feed for model's placeholder
  # feed is dictionary whose key is placeholder, and value is feeded value(numpy array)
  feed = {model.x: x_train, model.y: y_train}

  # Go training via running train_op with feeded data!
  # We run simultaneously train_op(backprop) and loss value
  _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

  # print loss stat every 100 iterations
  if i%100 == 0:
    print "| steps %07d | loss: %.3lf" % (i, loss)
