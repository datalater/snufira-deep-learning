
ⓒ JMC 2017

TensorFlow Practice

---

## 03 TensorFlow Basic ()

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
