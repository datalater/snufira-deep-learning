
ⓒ JMC 2017

TensorFlow Practice


---
## 04 MLP iris

### 개요

#### 0. 데이터를 가져온다

+ 코드0-1. dataset을 load 한다.
+ 코드0-2. dataset의 data와 target shape를 출력해본다.
+ 코드0-3. input 데이터와 label 데이터는 train : test = 7 : 3 으로 구성한다.

```python
import numpy as np
import tensorflow as tf
from sklearn import datasets, model_selection
import math

### define input data
# Load the iris dataset
iris = datasets.load_iris()

# iris has two attributes: data, target
print(iris.data.shape)
print(iris.target.shape)

# Split the data into training/testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(
  iris.data, iris.target, test_size=0.3)
```

#### 1. 모델(layer)을 build 한다

+ 코드1-1. layer의 hyperparameter를 정의한다.
+ 코드1-2. layer에 input 데이터를 넣기 위해 placeholder로 선언한다.
  + input 데이터 x는 column이 몇 개 필요한가?
  + input 데이터 y는 column이 몇 개 필요한가?
+ 코드1-3. 각 hidden layer를 with 절로 선언한다.
  + `with tf.name_scope('hidden1'):`
+ 코드2. layer의 parameter를 Variable로 선언한다. 초기값은 `truncated_normal([shape, shape], stddev=1.0)`로 넣는다.
  + hidden layer를 구성하는 parameter w와 b는 column이 몇 개 필요한가?
  + `weights1`
  + `biases1`
  + `hidden1`
+ 코드3-1. hidden layer의 방정식을 세운다. (using `tf.nn.relu`, `tf.matmul`)
  + linear function에 non-linear function을 씌운다.
  + `hidden1`
+ 코드3-2. 같은 방식으로 hidden layer를 1개 더 만든다.
  + `weights2`, `biases2`, `hidden2`
+ 코드3-3. 마지막 layer는 non-linear function을 씌우지 않는다.
  + `weights3`, `biases3`, `logits`
+ 코드3-4. logits 벡터에서 가장 큰 값의 index를 `argmax()`로 구하고 `int`형으로 `tf.cast` 한 후에 pred로 할당한다.
  + `pred`
+ 코드3-5. pred와 y_labe을 boolean으로 비교한 다음 `tf.reduce_mean`을 해서 accuracy를 구한다.
  + `accuracy`

```python
### define hyperparameters
n_classes = 3
n_features = 4
n_hidden_1 = 4
n_hidden_2 = 4

learning_rate = 0.01
max_iter = 100000
```

```python
### define graph
x = tf.placeholder(tf.float32, shape=[None, n_features], name="X")
y_label = tf.placeholder(tf.int32, shape=[None], name="Y_label")
# one-hot encoding
y = tf.one_hot(indices=y_label, depth=n_classes)
```

```python
# hidden 1
with tf.name_scope('hidden1'):
  weights1 = tf.Variable(
      tf.truncated_normal([n_features, n_hidden_1],
                          stddev=1.0),
      name='weights')
  biases1 = tf.Variable(
      tf.zeros([n_hidden_1]),
      name='biases')
  hidden1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
# hidden 2
with tf.name_scope('hidden2'):
  weights2 = tf.Variable(
      tf.truncated_normal([n_hidden_1, n_hidden_2],
                          stddev=1.0),
      name='weights')
  biases2 = tf.Variable(
      tf.zeros([n_hidden_2]),
      name='biases')
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)
# Linear
with tf.name_scope('softmax_linear'):
  weights3 = tf.Variable(
      tf.truncated_normal([n_hidden_2, n_classes],
                          stddev=1.0),
      name='weights')
  biases3 = tf.Variable(
      tf.zeros([n_classes]),
      name='biases')
  logits = tf.matmul(hidden2, weights3) + biases3

pred = tf.cast( tf.argmax(logits, 1), tf.int32 )
accuracy = tf.reduce_mean( tf.cast( tf.equal(pred, y_label), tf.float32 ))
```

#### 2. 모델의 loss를 계산한 후 minimize 한다

+ 코드4. loss를 `tf.reduce_sum`을 이용해서 최소제곱법으로 구한다. `loss = 1/2 * 1/N(sigma (f(x)-y)^2)`
  + `n_samples`
  + `loss`
+ 코드5. optimzer 알고리즘을 gradient descent로 사용하고 loss를 minimize 한다.
  + `learning_rate`
  + `train_step`

```python
n_samples = tf.cast(tf.size(x), tf.float32)
loss = tf.reduce_sum(tf.pow(y_h-y, 2))/(n_samples * 2)

# define optimization function
learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

#### 3. 모델을 Train 한다

+ 코드6. Session()을 with 절로 선언하고 Variable을 초기화하여 run 한다.
+ 코드7. train 반복 횟수를 정의한다.
  + `max_iter`
+ 코드8. train 반복 횟수만큼 반복문을 선언한다.
+ 코드9. w, b, y_h, loss의 value를 `sess.run()`으로 출력한다. fetch할 tensor를 첫 번째 인자로 넣고 두 번째 인자로 feed를 넣는다.
  + `_`
  + `v_w_val`
  + `v_b_val`
  + `y_h_val`
  + `loss_val`

+ 코드10. 1000번째 step마다 'Epoch'을 step 값으로, 'Loss'를 `loss_val`로 출력한다.
  + parameter를 update하는 한 단위를 Epoch이라고 한다.
+ 코드11. train이 끝난 이후에 paramter w를 `Coefficients`로 출력하고, 최종 모델의 "Mean squared error"를 `loss_val`로 출력한다.
+ 코드12. test 데이터를 black으로 scatter 한다.
+ 코드13. test 데이터에 대한 prediction을 하기 위해 feed와 출력할 tensor를 구성하고 `sess.run()`으로 구한다.
+ 코드14. test 데이터 x와 prediction을 파란색 line으로 plot한다.
+ 코드15. plot을 출력한다.
  + `plt.show()`

```python
### Starting sessions
with tf.Session() as sess:
  ## initialize variables
  init = tf.global_variables_initializer()
  sess.run(init)

  max_iter = 10000

  for i in range(max_iter):
    _, v_w_val, v_b_val, y_h_val, loss_val = sess.run(
      [train_step, v_weight, v_bias, y_h, loss],
      feed_dict={x: diabetes_X_train, y: diabetes_y_train})

    if i % 1000 == 0:
      print('Epoch ', i)
      print('Loss', loss_val)

    if math.isnan(loss_val):
      print('LOSS is NAN!')
      break

  # The coefficients
  print('Coefficients: \n', v_w_val)
  # The mean squared error
  print("Mean squared error: %.2f" % loss_val )


  # Plot outputs
  plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
  test_pred = sess.run(y_h,
    feed_dict={x: diabetes_X_test, y: []})
  plt.plot(diabetes_X_test, test_pred,
          color='blue', linewidth=3)

  plt.xticks(())
  plt.yticks(())

  plt.show()
```

---
---

## 03 TensorFlow Basic (with scikit-learn)

---

### 개요

#### 0. 데이터를 가져온다

+ 코드0-1. dataset을 load 한다.
+ 코드0-2. dataset의 data와 target shape를 출력해본다.
+ 코드0-3. 2번째 feature만 사용해서 input 데이터를 구성하고, label은 전체 target 데이터 모두를 사용한다.
+ 코드0-4. input 데이터와 label 데이터는 각각 전체 데이터에서 -20번째까지 train으로, 나머지는 test로 구성한다.

```python
#################
### Perceptron for linear regression
#################
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# diabetes has two attributes: data, target
print(diabetes.data.shape)
print(diabetes.target.shape)

# diabetes consists of 442 samples
#with 10 attributes and 1 real target value.

# Use only one feature
diabetes_X = diabetes.data[:, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
```

#### 1. 모델을 세운다

+ 모델
    + `h = wx + b`
+ 코드1. 모델에 input 데이터를 넣기 위해 placeholder로 선언한다.
  + input 데이터 x의 column은 몇 개 필요한가?
  + input 데이터 y의 column은 몇 개 필요한가?
+ 코드2. 모델의 parameter를 Variable로 선언한다. 초기값은 0으로 넣는다.
  + parameter w와 b의 column은 몇 개 필요한가?
  + `v_weight`
  + `v_bias`
+ 코드3. 모델의 방정식을 세운다. (using `tf.add` and `tf.multiply`)
  + `y_h`

```python
x = tf.placeholder(tf.float32, shape=None, name='x-input')
y = tf.placeholder(tf.float32, shape=None, name='y-input')

v_weight = tf.Variable(
  0,
  dtype=tf.float32,
  name = "W")
v_bias = tf.Variable(
  0,
  dtype=tf.float32,
  name = "w0")

y_h = tf.add( tf.multiply(x, v_weight), v_bias )
```

#### 2. 모델의 loss를 계산한 후 minimize 한다

+ 코드4. loss를 `tf.reduce_sum`을 이용해서 최소제곱법으로 구한다. `loss = 1/2 * 1/N(sigma (f(x)-y)^2)`
  + `n_samples`
  + `loss`
+ 코드5. optimzer 알고리즘을 gradient descent로 사용하고 loss를 minimize 한다.
  + `learning_rate`
  + `train_step`

```python
n_samples = tf.cast(tf.size(x), tf.float32)
loss = tf.reduce_sum(tf.pow(y_h-y, 2))/(n_samples * 2)

# define optimization function
learning_rate = 0.1
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

#### 3. 모델을 Train 한다

+ 코드6. Session()을 with 절로 선언하고 Variable을 초기화하여 run 한다.
+ 코드7. train 반복 횟수를 정의한다.
  + `max_iter`
+ 코드8. train 반복 횟수만큼 반복문을 선언한다.
+ 코드9. w, b, y_h, loss의 value를 `sess.run()`으로 출력한다. fetch할 tensor를 첫 번째 인자로 넣고 두 번째 인자로 feed를 넣는다.
  + `_`
  + `v_w_val`
  + `v_b_val`
  + `y_h_val`
  + `loss_val`

+ 코드10. 1000번째 step마다 'Epoch'을 step 값으로, 'Loss'를 `loss_val`로 출력한다.
  + parameter를 update하는 한 단위를 Epoch이라고 한다.
+ 코드11. train이 끝난 이후에 paramter w를 `Coefficients`로 출력하고, 최종 모델의 "Mean squared error"를 `loss_val`로 출력한다.
+ 코드12. test 데이터를 black으로 scatter 한다.
+ 코드13. test 데이터에 대한 prediction을 하기 위해 feed와 출력할 tensor를 구성하고 `sess.run()`으로 구한다.
+ 코드14. test 데이터 x와 prediction을 파란색 line으로 plot한다.
+ 코드15. plot을 출력한다.
  + `plt.show()`

```python
### Starting sessions
with tf.Session() as sess:
  ## initialize variables
  init = tf.global_variables_initializer()
  sess.run(init)

  max_iter = 10000

  for i in range(max_iter):
    _, v_w_val, v_b_val, y_h_val, loss_val = sess.run(
      [train_step, v_weight, v_bias, y_h, loss],
      feed_dict={x: diabetes_X_train, y: diabetes_y_train})

    if i % 1000 == 0:
      print('Epoch ', i)
      print('Loss', loss_val)

    if math.isnan(loss_val):
      print('LOSS is NAN!')
      break

  # The coefficients
  print('Coefficients: \n', v_w_val)
  # The mean squared error
  print("Mean squared error: %.2f" % loss_val )


  # Plot outputs
  plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
  test_pred = sess.run(y_h,
    feed_dict={x: diabetes_X_test, y: []})
  plt.plot(diabetes_X_test, test_pred,
          color='blue', linewidth=3)

  plt.xticks(())
  plt.yticks(())

  plt.show()
```

---
---

## 02 TensorFlow Basic (MNIST)

---

### 개요

#### 0. 데이터의 차원을 계산한다

+ input 데이터 : 28 by 28 이미지이므로 784차원이다.
+ prediction : class가 총 10개이므로 10차원이다.

#### 1. 모델을 세운다

+ 모델
  + `logit = wx + b`
  + `prediction = softmax(logit)`

+ 코드1. 모델에 input 데이터를 넣기 위해 placeholder로 선언한다.
+ 코드2. 모델의 parameter를 Variable로 선언한다.
+ 코드3. 모델에 parameter와 input 데이터를 넣어서 parameter가 input 데이터의 정보를 갖고 있게 한 후, 모델의 prediction의 값을 확률 값으로 바꾼다.

#### 2. 모델의 cost를 계산한 후 minimize 한다

+ 코드4. 모델의 prediction(=`logits`)과 true label(=`onehot_labels`) 사이의 cost를 cross entropy로 구한다.
+ 코드5. optimzer 알고리즘을 사용해서 cost를 minimize 한다.

#### 3. 모델의 accurracy를 정의하기 위해 onehot label과 prediction을 비교한다

+ 코드6. predictions 벡터와 labels 벡터를 대상으로 각 row마다 값이 가장 큰 column의 index를 구한다.
+ 코드7. 구한 index의 값들이 서로 같은지 boolean으로 비교한다.
+ 코드8. 숫자로 표현된 boolean 벡터의 평균을 계산하여 accuracy를 구한다.

#### 4. 모델을 Train 한다

+ 코드9. train_step을 정의한다.
+ 코드10. Session()을 with 절로 선언하고 Variable을 초기화하여 run 한다.
+ 코드11. train_step 만큼 반복문을 선언하고 batch 단위로 input 데이터와 label을 가져온다.
+ 코드12. feed를 정의한 다음, `sess.run()`에 fetch할 tensor를 첫 번째 인자로 넣고 두 번째 인자로 feed를 넣는다.


---

### 코드1. 모델의 input 데이터를 placeholder로 선언한다.

+ MNIST input 데이터 : 784 차원

```python
model_inputs = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None,10])
```

> **Note:** session runtime 때 동적으로 tensor의 값을 주입(feed)할 데이터는 shape과 dtype을 고려하여 placeholder로 선언한다.

### 코드2. 모델의 parameter를 Variable로 선언한다.

+ MNIST input 데이터 : 784 차원

```python
w = tf.Variable(tf.random_normal(shape=[784, 10]))
b = tf.Variable(tf.random_normal(shape=[10]))
```

> **Note:** 학습하고자 하는 모델의 parameter는 값이 변경되어야 하므로 Variable로 선언한다.

### 코드3. 모델의 parameter에 input 데이터의 정보를 곱셈으로 넣고, 모델의 prediction의 값을 확률 값으로 바꾼다.

+ MNIST

```python
logits = tf.matmul(model_inputs, w) + b
predictions = tf.nn.softmax(logits)
```

> **Note1:** 모델의 parameter에 input 데이터를 곱하는 것은 일종에 parameter와 input 데이터를 결합하는 방식 중 하나이다. 모델은 training data를 학습한다. 학습한다는 말은 모델이 training data의 정보를 잘 담고 있다는 뜻이다. 학습이 끝나면 모델의 parameter는 모든 training data와 곱해져서 결국 고정된 w값을 갖게 된다. 이때 w값은 training data의 정보 전부를 담고 있다.
> **Note2:** 덧셈하는 b는 bias를 뜻한다. bias는 training data와 결합되지 않고, 모델의 output에 대한 data indenpendent preference를 부여한다. 예를 들어, training data set이 unblanced 되어 cat에 대한 이미지가 dog보다 훨씬 많다면, cat에 대한 bias는 다른 class보다 더 높을 것이다.
> **Note3:** parameter에 input 데이터의 정보를 곱셈으로 넣어서 bias를 더하는 방식은 linear 함수의 방식이다. linear function은 linear function끼리 자유롭게 결합할 수 있는 장점이 있다.
> **Note4:** linear function의 강력함을 표현하기 위해, linear function을 레고 블럭으로 비유한다. 레고 블럭은 블럭을 자유롭게 결합해서 다양한 결과물을 만들어 낸다. 이러한 특성 때문에 뉴럴 네트워크는 linear function을 사용한다. 뉴럴 네트워크는 여러 layer를 사용해서 다양한 연산을 여러 번 거치는 구조이기 때문이다.

### 코드4. 모델의 predictions(=`logits`)와 true labels(=`onehot_labels`) 사이의 loss를 cross entropy로 구한다.

+ cross entropy

```python
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predictions)
```

### 코드5. optimzer 알고리즘을 정의해서 cost를 minimize 한다.

+ gradient descent

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)
```

> **Note:** optimizer는 모델이 loss를 최소화하는 방향으로 parameter를 업데이트하도록 만든다.

### Accuracy 코드 6~8

```python
코드6. predictions 벡터와 labels 벡터를 대상으로 각 row마다 값이 가장 큰 column의 index를 구한다.
코드7. 구한 index의 값들이 서로 같은지 boolean으로 비교한다.
코드8. 숫자로 표현된 boolean 벡터의 평균을 계산하여 accuracy를 구한다.
```

```python
dense_predictions = tf.argmax(predictions, axis=1)
dense_labels = tf.argmax(labels, axis=1)
equals = tf.cast(tf.equal(dense_predictions, dense_labels), tf.float32)
accuracy = tf.reduce_mean(equals)
```

> **Note:** `argmax()`에서 axis는 tensor의 shape의 index를 의미한다. 예를 들어, `prediction.shape`가 `(None, 10)`이라면 `None`이 `axis=0`, `10`이 `axis=1`이 된다. tensor가 2차원 벡터일 때 `axis=0`으로 argmax를 구하면 각 column마다 가장 큰 row의 index를 구한다. `axis=1`로 armax를 구하면 각 row마다 가장 큰 column의 index를 구한다.
> **Note:** predictions의 shape는 `(None, 10)`이다. 우리가 원하는 것은 각 row마다 가장 큰 colulmn의 index를 구하는 것이기 때문에 `axis=1`로 설정한다.

### Train 코드 9~13

```
코드9. train_step을 정의한다.
코드10. Session()을 with 절로 선언하고 Variable을 초기화하여 run 한다.
코드11. train_step 만큼 반복문을 선언하고 batch 단위로 input 데이터와 label을 가져온다.
코드12. feed를 정의한 다음, `sess.run()`에 fetch할 tensor를 첫 번째 인자로 넣고 두 번째 인자로 feed를 넣는다.
코드13. step, loss, accuracy를 print한다.
```

```python
train_step = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(train_step):
        batch_inputs, batch_labels = mnist.train.next_batch(100)
        images_val, labels_val = mnist.validation.next_batch(100)
        feed = {model_inputs : batch_inputs, labels : batch_labels}
        _, loss_val = sess.run([train_op, loss], feed_dict=feed)
        # loss_val = sess.run(loss, feed_dict=feed)

        if step % 250 == 0:
            print ("step {} | loss : {}".format(step, loss_val))

        # feed = {model_inputs : images_val, labels : labels_val}
        # accuracy = sess.run(accuracy, feed_dict=feed)
        # print ("acc : {}".format(accuracy))
```

> **Note:** `sess.run` 하는 그래프에 `tf.Variable`이 하나라도 포함되어 있을 경우, 해당 변수를 반드시 초기화해야 한다.
> **Note:** `sess.run(fetch)` : sess.run()으로 출력할 tensor를 인자로 넣는다.


---
---

## 01 TensorFlow Basic (linear regression)

---

### 개요

#### 1. 모델을 세운다 `y = wx + b`

+ 코드1. 모델의 parameter를 Variable로 선언한다.
+ 코드2. 모델에 데이터를 넣기 위해 placeholder로 선언한다.
+ 코드3. 모델 방정식을 세운다.

#### 2. 모델의 cost를 계산한 후 minimize 한다.

+ 코드4. cost를 최소제곱법으로 구한다.
+ 코드5. optimzer 알고리즘을 사용해서 cost를 minimize 한다.

#### 3. tensor를 모두 연결하기 위해 세션으로 graph를 그린다.

+ 코드6. Session을 선언하고 Variable을 초기화하여 run 한다.

#### 4. 만들어 둔 graph에 데이터를 feed하여 train을 반복한다.

+ 코드7. 데이터를 feed하여 session을 run한다.

---

### 코드1. 모델의 input 데이터를 placeholder로 선언한다.

+ feature가 1개일 때

```python
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
```

+ feature가 3개일 때

```python
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
```

+ MNIST 784 차원

```python
model_inputs = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None,10])
```

> **Note:** session runtime 때 동적으로 tensor의 값을 주입(feed)할 데이터는 shape과 dtype을 고려하여 placeholder로 선언한다.

### 코드2. 모델의 parameter를 Variable로 선언한다.

+ feautre가 1개일 때

```python
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```

+ feature가 3개일 때

```python
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```

+ MNIST 784 차원

```python
w = tf.Variable(tf.random_normal(shape=[784, 10]))
b = tf.Variable(tf.random_normal(shape=[10]))
```

> **Note:** 학습하고자 하는 모델의 parameter는 값이 변경되어야 하므로 Variable로 선언한다.

### 코드3. 모델 방정식을 세운다.

+ linear regression 모델일 때

```python
hypothesis = X * W + b
```

### 코드4. 모델의 prediction과 true label 사이의 cost를 구한다.

+ 최소제곱법 (MSE)

```python
cost = tf.reduce_mean(tf.square(hypothesis - Y))
```

### 코드5. optimzer 알고리즘을 정의해서 cost를 minimize 한다.

+ gradient descent

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```

### 코드6. Session을 선언하고 Variable을 초기화하여 run 한다.

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

> **Note:** `sess.run` 하는 그래프에 `tf.Variable`이 하나라도 포함되어 있을 경우, 해당 변수를 반드시 초기화해야 한다.

### 코드7. 데이터를 feed하여 session을 run한다.

```python
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
```

---
