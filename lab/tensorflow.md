
ⓒ JMC 2017

TensorFlow Practice

---

---

## linear_regression_feed.py (feature 1개)

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

+ 코드6. Session을 선언하고 모든 변수를 초기화하여 run 한다.

#### 4. 만들어 둔 graph에 데이터를 feed하여 train을 반복한다.

+ 코드7. 데이터를 feed하여 session을 run한다.

---

### 코드1. 모델에 데이터를 넣기 위해 placeholder로 선언한다.

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

### 코드3. 모델 방정식을 세운다.

```python
hypothesis = X * W + b
```

### 코드4. cost를 구한다. ex) 최소제곱법

```python
cost = tf.reduce_mean(tf.square(hypothesis - Y))
```

### 코드5. optimzer 알고리즘을 사용해서 cost를 minimize 한다.

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```

### 코드6. Session을 선언하고 모든 변수를 초기화하여 run 한다.

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

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
