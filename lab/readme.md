

JMC 2017

**Tensorflow API Guide**  
\- "https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer"

---

## Kaggle: Titanic




---

## RNN2 review

### 실습 개요

+ character-level language modeling with Korean novel
+ 구운몽 텍스트를 학습하여 다음에 나올 character를 예측하여 소설을 생성하는 RNN 모델을 만든다.

> **Note**: 이번 실습에서 다루는 한글의 character-level은 모든 자모 단위이다. ex. 구 -> {ㄱ,ㅜ}

### Data pre-processing :: step(1) :: characcter 단위

+ 한글 파일(구운몽 텍스트)을 읽어들이고, `character 단위`로 나눈다.
  + 한글 파일의 인코딩을 다르게 하면 자동으로 나눠지게 된다. (한 글자를 구분하는 단위로 낙지처럼 생긴 문자가 생성됨)

### Data pre-processing :: step(2) :: vocabulary dictionary

+ 나눠진 각각의 character를 unicode로 변환하여 `vocabulary dictionary`를 구성한다.
  + ex. ㅎ -> u'\u3160' -> 0
  + ex. {'ㅎ':0}

### Data pre-processing :: step(3) :: training input x

+ 구성한 vocabulary를 기반으로 구운몽 텍스트를 character index로 모두 변환하여 `training input x`를 얻는다.

### Data pre-processing :: step(4) :: ground-truth y

+ training input x로부터 `ground-truth y`를 구성한다.
  + 우리의 목표는 x_0, ..., x_{t-1}를 받아 다음 character인 x_t를 예측하는 것이다.
  + 따라서 y_t = x_{t+1}로 세팅하면 된다.
  + 코드 관점에서 보면 X[n]을 Y[n+1]로 대응시키면 된다.
  + 실습 코드에서 Y[0]은 X[-1]로 대입했는데, 다른 값으로 해도 크게 상관없다고 함.

### Data pre-processing :: step(5) batch 단위

+ batch 단위로 training 해야 하므로, x와 y를 `batch 단위`로 나눠놓는다.
  + 만약 batch 단위로 training 하지 않고, 전체 데이터를 한꺼번에 training 하게 되면,
  + sequence length가 무진장 길어지게 된다.
  + 즉, 데이터의 양이 무진장 길어지게 된다는 것이다.
  + 그런데 hidden state의 개수는 우리가 이미 정해주었다. (ex. rnn_size = 128)
  + 즉, 많은 양의 데이터(=정보)를 representation하기에 hidden state 개수가 고정되어 있으므로 부적합하다.

> **Note:** 그러면 많은 양의 데이터를 representation 할 수 있도록 hidden_state의 개수를 크게 늘리면 어떨까?  
> 좋지 않다. 왜냐하면 hidden_state가 커지는 만큼 parameter가 많아지고, parameter가 너무 많으면 optimization이 힘들어지기 때문이다.

> **Note:** 그래서 RNN 류의 네트워크 (vanila RNN, GRU RNN)는 long sequence data를 처리하기에는 부적합하다는 논의가 현재 이어지고 있다.

#### 개념 구분하기 :: training example / batch_size / num_batches / sequence_length

+ sequence_length : 하나의 sequence에 포함되는 token의 개수
+ example : sequence 1개
+ batch_size : example의 개수 (=sequence의 개수)

---

+ 전체 데이터 = [1, 55, 32, 11, ....]

> **Note:** 전체 데이터를 구성하고 있는 index는 각각의 character를 뜻한다. 아래 설명에서 chacter와 index를 같은 의미로 썼다.

+ example = [c1, c2, c3, ..., c100]
  + 여기서 example은 1개의 example을 뜻한다.
  + example은 전체 데이터 중 일부 character를 추출한 것이다.
  + example에 포함된 chacter의 개수를 정하는 것이 seq_length(sequence length)이다.
+ batch_size = 32
  + batch size란 하나의 batch에 몇 개의 example을 넣을 것인지를 말한다.
  + batch_size =32 라면 100개의 character를 담고 있는 example을 32개 뽑은 것이다.
  + 즉, 하나의 batch에는 character 3,200개가 들어간다.

> **Note:** 하나의 example에 포함되는 token의 개수 = sequence_length

+ num_batches = int( 전체데이터.size / (batch_size * seq_length) )
  + if,
  + 전체데이터.size = 500,000
  + batch_size = 32 (hyperparameter)
  + seq_length = 100 (hyperparameter)
  + num_batches = int(500,000 / (3,200))

### 모델링 :: step(6-1) :: TensorFlow Graph :: x_i, y_i

+ x_i = (batch_size, seq_len)
+ y_i = (batch_size, seq_len)

```python
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets    = tf.placeholder(tf.int32, [batch_size, seq_length])
```

### 모델링 :: step(6-2) :: TensorFlow Graph :: embedding

+ x_i를 character embedding space d로 projection 하고, RNN의 input에 맞게 shape를 조정한다.
  + character embedding space : vocabulary_dictionary에 저장되어 있는 key 값의 개수만큼의 차원을 갖고 있는 space

---

## RNN2 20170830 나세일 조교 실습

### MLP p.5

+ MLP란 affine transformation이다.
+ `affine transformation이란 선형함수를 만드는 것`이다.
+ 슬라이드에 나와 있는 것이 MLP의 기본 구조
+ 딥러닝에서 affine transformation과 같은 의미로 통용되는 것
  + fully-connected layer
  + linear transformation
  + projection

### MLP p.6

+ 강조 : '마지막 layer는 ...'

### MLP p.7

+ 첫번째 layer의 차원을 d차원으로 정한다. (연구자가 정하는 hyperparameter)
+ 첫번째 layer는 input의 차원인 h * d차원

### p.8

+ input size가 달라지면 parameter size도 달라진다.
+ 작은 이미지를 학습시키면 parameter size가 작으면 되는데
+ 큰 이미지를 학습시키면 parameter size가 커야 한다.
+ 즉, training exmaple마다 parameter size가 달라진다.
+ 우리가 원하는 것은 이게 아님. 구현 상의 문제점도 있고.
+ `모델의 parameter 수는 input size와 관계없이 항상 고정되어야 한다`.

### p.9

+ 이것을 해결하는 방법
+ input size를 고정하기 위해 이미지의 크기를 같은 사이즈로 resize 또는 crop 해준다.
+ 보통 이미지 작업 때 해주는 작업.

### p.10

+ 데이터 소스가 이미지가 아니라 텍스트라면?
+ 텍스트의 길이를 똑같이 맞춰주려고 한다면, 정보의 손실이 많아진다.
  + ex. I am a boy. -> I am
+ 이미지는 사이즈를 바꿔도 정보의 손실이 많지 않다.
+ MLP℠s training data가 다른 것, 즉 variable length data를 제대로 학습할 수가 없다.
+ 그런데 RNN은 variable length data를 제대로 핸들링 할 수 있다.
+ `RNN은 가변 길이 데이터를 처리할 수 있기 때문에 사용한다`.

### ṗ.11

+ RNN의 구성 : input 과 hidden state
+ input은 time step 단위로 나눠진다.
+ time step으로 나눈 데이터를 tokenized data라고 한다.
  + ex. "I am a boy"에서 첫 번째 token은 "I"가 된다.
+ time step마다 hidden state가 update 된다.

### p.12

+ $h_t = f(x_t, h_{t-1})$
  + 이전 hidden state와 현재 state의 input 데이터를 받는다.
+ hidden state가 갖는 정보 :
  + h0 : 첫번째 token에 대한 정보
  + h1 : 첫번째 hidden state(=첫번째 input에 대한 정보)와 두번째 input을 받는다.
  + h_t : t번째까지의 모든 input에 대한 정보를 담고 있음
+ 순서가 유의미한 데이터를 핸들링할 때 RNN이 많이 쓰인다.

### p.13

+ 현재 상태의 input을 W1을 통해 projection하고
+ 이전 상태의 hidden state를 W2를 통해 projection 하고
+ b를 더한 후
+ 전체에 대해 hyper tangent 함수(activation function)를 적용한다.

### p.14

+ 각 time step이 가지는 hidden state의 의미를 알았으니 그림을 살펴보면 이해할 수 있을 것이다.
+ image captioning
  + 주어진 image를 잘 설명하는 자연어 text를 생성하는 것
  + input은 고정 크기
  + 이미지를 설명하는 text는 이미지마다 달라지므로 RNN을 사용한다.
+ video classification
  + 마지막 hidden state만을 사용해서 결과를 낸다.
  + 처음부터 끝까지 다 본 후에 operation 하는 것
+ machine translation
  + input의 마지막 hidden state(한국어를 한 문장을 잘 설명하는)로 시작해서 영어로 번역한다.
+ Language modeling
  + 우리가 실습 때 할 것

### p.16~17 LM

+ token을 어떤 단위로 나누느냐에 따라 LM을 다르게 부른다.
+ show me가 token으로 온 후에 다음에 뭐가 올 것인지 예측하는 게 word-level Language Modeling


### p.19 한국어 character-level LM

+ 한국어는 영어와 달리 무엇을 character로 볼지 애매함
+ 이번 실습에서는 한글을 자모단위로 나눠서, 각각의 자음과 모음을 하나의 character로 볼 것임
+ 첫번째 token 다음에 나오는 token, 그것 다음에 나오는 token, 다음에 나오는 token... 다음 token.. 이런 식으로 generate하는 모델을 만들 것임

---

## 실습

+ 우리가 사용할 데이터
+ 구운몽 한글 파일 `http://localhost:8888/edit/data/nine_dreams/ninedreams_utf8.txt`
+ 우리가 사용할 python 파일
  + `http://localhost:8888/notebooks/4_kor_char_rnn_train.ipynb`
  + `http://localhost:8888/edit/TextLoader.py`
  + `http://localhost:8888/notebooks/4_kor_char_rnn_inference.ipynb`

### pickle (python)

+ pickle : 파이썬 내부에 쓰던 오브젝트를 디스크에 저장해놓는다.
+ pickle을 이용하면 파이썬 파일로 작성한 변수를 작업 메모리에 올려놓을 수 있다.

```python
a = 1
b = [12,3,2,42]
c = {"dsf":"sdfs", "qwe": "sdfsa"}

with open("b_value.pkl", "w") asf:
    pickle.dump(b, f)
print a, b, c

# 아래처럼 딕셔너리로 모든 오브젝트를 한번에 넣어두면 편하게 사용할 수 있다. 튜플은 순서가 바뀌므로 딕셔너리가 편리하다.
total = {}
total["a_value"] = a
total["b_value"] = b
pickle.dump(total, f)
```

+ 데이터 pre-processing한 것을 저장해 둔 다음에 training할 때 pickle을 많이 사용한다.

### p.19~ 코드 설명 (rec 1:07:00)


### p.22

+ X : 구운몽 소설을 character의 index로 바꾼 것
+ Y : X에 대응되는 매칭되는 character
  + X의 0에 대응되는 character는 Y에서 1
  + X의 0, 1, 2에 대응되는 character는 Y에서 32
  + 우리가 예측한 값과 실제 Y의 차이를 Loss로 구성한다.
+ $y_t= x_{t+1}$ Y는 X를 한 칸씩 밀어버리면 된다.

### p.24

+ pair 하나가, 하나의 example
+ 그 example을 batch size 만큼 모아두는 게, 하나의 batch

### p.25 step(6-1) 모델링

+ 이전까지는 TextLoader.py, 즉 pre-processing 이었음

### p.26 step(6-2)

+ $x_i$ = (b, l) # batch * sequence_length
+ x[3, 4] = [ 0, 0, 0, .., 1, 0, 0] # 3번째 batch의 3번째 token
+ x[3, 4] = 43 (지금은 이렇게 되어 있음)
+ vocab_size = v
+ x[3, 4] = W X (v) = d
  + W : embedding matrix
  + v : one_hot vector
  + W의 각 i열은, v의 i번째 index에 해당하는 캐릭터를 나타내는 임베딩이라고 한다.
  + W의 값을 학습할 때도 backprop해서 학습한다.
  + 임베딩 매트릭스의 각 칼럼은, 각 캐릭터의 의미를 유의미하게 캐치하는 칼럼 벡터이다.

> **Note:** 단어마다 embedding matrix가 있는 것이 word2vector. ex) man - woman + king = queen

+ index = (d) dimension
+ (index) = seq_len # seq_len 만큼 index 개수가 있다.
+ batch * seq_len = (batch * seq_len * d)

### Q. RNN size

+ RNN의 구성요소
  + x_t = (k)
  + h = (d) = [0, 0, 0, ..., 0] (h0 초기화)
    + h1 = w1*x_t + w2*h0 + b
      + w1*x_t = (d)
      + w2*h0 = (d)
      + b = (d)
    + h2 = w1*x1 + w2*h1 + b
  + w1 = (d, k)
  + w2 = (d, d)
  + b = (d)

---

+ h의 dimension이 커질수록 input 정보를 담을 수 있는 양이 커지는 것이다.
+ 그런데 문제는 h의 dimension이 너무 크면 담고 있는 정보가 너무 많아서 optimization이 힘들어진다.
+ parameter가 너무 많으면 optimization이 힘들어지기 때문이다.

---

### p.27 step(6-3)

+ rec 2:23:00

### p.29 step(6-5)

+ 파란색 박스에 있는 input을 RNN에 태운 다음
+ h_t = rrn_size
+ softmax_w = (rnn_size, vocab_size)
+ prob = (vocab_size)
+ [0.01, 0.3, 0.2, 0.1, ...]  # softmax 취한 결과, 각 다음 단어가 나올 확률 분포

### inference

+ ckpt에서 들고 온 것
  + embedding
  + RNN
  + softmax_w(b)
+

### RNN이 생성해낸 글의 의미

+ rec 2:41:00

---
---

## Lecture note

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
