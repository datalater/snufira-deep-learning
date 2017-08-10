

JMC 2017

**Tensorflow API Guide**  
\- "https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer"


---

### 01 Tensor : 데이터 저장의 기본 단위이며 값과 차원(형태)을 가진다.

```python
a = tf.constant(1.0, dtype=tf.float32, name='layer1_w') # 1.0의 값을 갖는 스칼라 tensor 생성
b = tf.constant(1.0, shape=[3,4], name='layer1_b') # 1.0의 값을 갖는 3x4 2차원 tensor 생성
c = tf.constant(1.0, shape=[3,4,5]) # 1.0의 값을 갖는 3x4x5 3차원 tensor 생성
d = tf.random_normal(shape=[3,4,5]) # normal distribution에서 3x4x5 3차원 tensor를 sampling

print(c) # Tensor("Const_4:0", shape=(3, 4, 5), dtype=float32)
```

+ `Const_4:` : 4번째 저장된 Constant

### 01 TensorFlow Programming의 개념

+ (1) `tf.placeholder` 또는 input tensor를 정의하여 input node를 구성한다.
+ (2) input node에서부터 output node까지 이어지는 관계를 정의하여 graph를 그린다.
+ (3) session을 이용하여 input node(tf.placeholder)에 값을 feed하고, graph를 run시킨다.

```python
a = tf.constant(1.0, shape=[2,3])
sess = tf.Session()
print(sess.run(a))  

# output
[[1 1 1]
 [1 1 1]]
```

+ tensor의 값을 출력하려면 node로 구성된 graph를 run시켜야 하므로 session을 활용해야 한다.

### 01 Tensor name

```python
a = tf.constant(1.0, dtype=tf.float32, name='layer1_w') # 1.0의 값을 갖는 스칼라 tensor 생성
b = tf.constant(1.0, shape=[3,4], name='layer1_b') # 1.0의 값을 갖는 3x4 2차원 tensor 생성
c = tf.constant(1.0, shape=[3,4,5]) # 1.0의 값을 갖는 3x4x5 3차원 tensor 생성
d = tf.random_normal(shape=[3,4,5]) # normal distribution에서 3x4x5 3차원 tensor를 sampling

print(c) # Tensor("Const_4:0", shape=(3, 4, 5), dtype=float32)
```

+ 모든 tensor는 name 인자를 가질 수 있다.
+ name 인자를 넣으면 원할 때마다 특정 tensor를 그룹핑, 저장, 수정, 재사용 등이 매우 간편해진다.
+ ex. `layer1_*` : * (와일드카드)를 사용하여 layer1_이하로 시작하는 모든 tensor의 name을 일괄적으로 지칭한다.

> **Note:** op : 함수, constant()도 일종의 함수이므로 op에 해당한다.

### 01 Placeholder에 원하는 값 feed하기

```
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
c = a+b

with tf.Session () as sess:
    feed = {a:1, b:2}
    print(sess.run(c, feed_dict=feed))
    feed = {a:2, b:4}
    print(sess.run(c, feed_dict=feed))
```

@@@resume: Quiz0

---

### 02 MNIST using Logistic Regression

> **Note:** MNIST : [엠니스트]

#### 1. 모델의 input 및 parameter 정의

```python
def main(_):
  mnist = input_data.read_data_sets("./data", one_hot=True)

  # define model input: image and ground-truth label
  model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784]) # 28x28 이미지이므로 784차원 이미지로
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])
```

+ `model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])`
  + 모델에 입력할 이미지가 28*28이므로 784차원 벡터로 선형 변환한다.
  + 그리고 batch(샘플 수)단위로 학습을 할 텐데 아직 batch 값을 안 정했으므로 None으로 남겨둔다.
  + 즉 하나의 행을 하나의 이미지로 취급하는 것이다.
  + `model_inputs`의 행렬 형태는 칼럼벡터 (그림1, 그림2, ... 그림n)이 되며 칼럼의 수(n)은 batch값이 된다.
+ `labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])`
  + 각 이미지는 10개의 클래스로 레이블링된다.
  + 즉 하나의 행을 각 그림에 해당하는 label 벡터로 취급한다. ex) 2번째 class일 경우 : [0.0, 1.0, 0.0, ... 0]
  + `label`의 행렬 형태는 칼럼벡터 (그림i의 클래스 값은 1.0이고 나머지는 0.0인 10차원 벡터)가 되며 칼럼의 수(n)은 batch값이 된다.

```python
  # define parameters for Logistic Regression model
  w = tf.Variable(tf.zeros(shape=[784, 10]))
  b = tf.Variable(tf.zeros(shape=[10]))

  logits = tf.matmul(model_inputs, w) + b
  predictions = tf.nn.softmax(logits)
```

+ `w = tf.Variable(tf.zeros(shape=[784, 10]))`
  + `w` : parameter로써, 모든 그림의 label을 일괄적으로 매칭시키는 constant한 값이다.
  + `w`는 한 로우에 10개의 값을 갖고 있고 로우의 수는 784개이다.
    + 784개인 이유는 그림이 784차원 벡터이기 때문이다.
    + 10개인 이유는 784차원 그림과 784차원 parameter를 10개씩 곱한 결과과 label의 형태와 일치해야 하기 때문이다.
  + `w`는 각 그림을 의미하는 784차원 로우벡터 * 784개 칼럼벡터를 곱한 결과 값이 label이 되도록 만드는 constant한 값이다.
    + 그림1 * p1 + 그림1 * p2 + ... + 그림1 * p10 = 그림1의 label
    + p1은 784차원 칼럼벡터이다.
  + 의미적으로 본다면 모든 그림을 각가의 label에 매칭시키는 만능 paramter 값을 찾으려는 것이다.
+ `b = tf.Variable(tf.zeros(shape=[10]))`
  + `b` : bias로써, 모든 그림에 쓸 수 있는 constant한 값이다.
+ `logits = tf.matmul(model_inputs, w) + b`
  + `logit`의 행렬 형태는 batch * 10이다.
  + 즉 10차원 로우벡터로 구성되어 있는데, 이때 10개의 값은 실수이다. ex) [2 -10 3 0 ....]
+ `predictions = tf.nn.softmax(logits)`
  + `softmax` : 소프트맥스 알고리즘은 위 logit을 구성하는 10차원 로우벡터의 값을 확률적 표현으로 바꿔준다.
  + ex) [2 -10 3 0 ....]을 보면 -10은 굉장히 작은 값이다. 즉 해당 그림이 2번째 클래스일 확률은 매우 낮다.
  + 그러나 이러한 실수값으로는 해석하기가 불편하므로 이를 확률적 표현으로 바꿔준다.
  + softmax 알고리즘을 거친 후는 10차원 로우벡터의 값이 확률적 표현으로 변환되므로 모두 더하면 1이 된다.
  + ex) [0.1 0.001 0.2 ...]

#### 2. loss 및 optimizer 정의

```python
# define cross entropy loss term
loss = tf.losses.softmax_cross_entropy(
  onehot_labels=labels,
  logits=predictions)
```

 + `cross entropy`
   + 예를 들어 스페이스 상에 두 점 간의 거리는 유클리디안 거리로 계산할 수 있듯이, 여기서는 label과 prediction의 확률분포의 차이를 계산하려 한다.
   + label과 prediction의 확률분포의 차이를 entropy라고 한다.

```python
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train_op = optimizer.minimize(loss)
```

+ `optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)`
  + tensorflow에서 gradient를 생각할 때 스페이스로 떠올리는 것은 복잡하고 어려우므로 부적절하다.
  + 여기서 gradient는 parameter를 얼만큼 변화시켰을 때 loss가 얼마나 변화하는지 측정하는 변화량이다. (dloss/dw)

#### 3. Training

```python
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(10000):
    batch_images, batch_labels = mnist.train.next_batch(100)
    feed = {model_inputs: batch_images, labels: batch_labels}
    _, loss_val = sess.run([train_op, loss], feed_dict=feed)
    print ("step {}| loss : {}".format(step, loss_val))
```

+ Session을 이용하여 `variable`을 초기화시켜준 후
+ 각 iteration마다 image와 label을 batch(샘플 수)단위로 가져오고
+ 가져온 데이터를 이용하여 feed를 구성한다.
+ `train_op`(가져온 데이터에 대한 loss를 최소화하도록 paramter를 업데이트하는 op(함수))

+ `_, loss_val = sess.run([train_op, loss], feed_dict=feed)`
  + paramter를 업데이트할 때마다 loss의 값이 어떻게 되는지 궁금하므로 그 값을 loss_val이라는 변수에 할당한다.

#### Quiz1. Accuracy Tensor 만들기

```python
  dense_predictions = tf.argmax(predictions, axis=1)
  dense_labels = tf.argmax(labels, axis=1)
  equals = tf.cast(tf.equal(dense_predictions, dense_labels), tf.float32)
  acc = tf.reduce_mean(equals)
```

+ `#` of label / batch = Accuracy

@@@resume

---










---
