

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
