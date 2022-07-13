### MNIST 분류 모델 만들기 - 신경망

```
from keras.datasets import mnist
from keras.utils import np_utils
```

```
import numpy
import sys
import tensorflow as tf
```

```
# 처음 다운일 경우, 데이터 다운로드 시간이 걸릴 수 있음. 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 5, figsize=(18,12) )
```

```
print("label={}".format(y_train[0:15]))   # x데이터 0~14개 가져오기
plt.gray()
for image, ax in zip( X_train, axes.ravel() ):
    ax.imshow(image) # 이미지 표시
```

```
X_train = X_train.reshape(X_train.shape[0],784)   # 60000, 28, 28 -> 60000, 784로 변경
```

```
# 데이터 값의 범위 0~255 -> 0~1 
X_train.astype('float64') 
X_train = X_train/255
import numpy as np
```

```
print(X_train.shape)               # 데이터 크기
print("데이터의 최대, 최소 :", np.min(X_train), np.max(X_train) )
# 테스트 데이터 전처리
X_test = X_test.reshape(X_test.shape[0],784)
X_test.astype('float64')
X_test = X_test/255
# OneHotEncoding - 10진수의 값을 0, 1의 값을 갖는 벡터로 표현
y_train_1D = np_utils.to_categorical(y_train, 10)
y_test_1D = np_utils.to_categorical(y_test, 10)
y_train[0:4]
y_train_1D[0:4]
from keras.models import Sequential
from keras.layers import Dense
m = Sequential()
m.add(Dense(512,input_dim=784, activation='relu'))
m.add(Dense(128, activation='relu') )
m.add(Dense(10,activation='softmax'))  #softmax
m.compile(loss="categorical_crossentropy", 
         optimizer='adam',
         metrics=['accuracy'])
```
```
### 배치 사이즈 200, epochs 30회 실행,
history = m.fit(X_train, y_train_1D, validation_data=(X_test, y_test_1D),
                epochs=30,
                batch_size=200,
                verbose=1)
```
```
### EarlyStopping()
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
np.random.seed(3)
```

```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

```
# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
y_val = y_train[50000:]
X_train = X_train[:50000]
y_train = y_train[:50000]
X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
'''
'''
# 훈련셋, 검증셋 고르기
train_rand_idxs = np.random.choice(50000, 10000)
val_rand_idxs = np.random.choice(10000, 5000)
```

```
X_train = X_train[train_rand_idxs]
y_train = y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
y_val = y_val[val_rand_idxs]
'''

'''
# 라벨링 전환
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
m = Sequential()
m.add(Dense(512, input_dim=784, activation='relu'))
m.add(Dense(128, activation='relu') )
m.add(Dense(10, activation='softmax') )
m.compile(loss="categorical_crossentropy",
          optimizer='adam',
          metrics=['accuracy'] )
m.summary()
```

```
from keras.callbacks import EarlyStopping
```

```
early_stopping = EarlyStopping(patience=30, monitor='val_loss')
```

```
hist = m.fit(X_train, y_train,
             epochs=3000,
             batch_size=10,
             validation_data=(X_val, y_val),
             callbacks=[early_stopping])
```

'''
hist = m.fit(X_train, y_train,
             epochs=3000,
             batch_size=10,
             validation_data=(X_val, y_val),
             callbacks=[early_stopping])
10:43
Shapes (10, 1) and (10, 10) are incompatible
```

```
# 5. 모델 학습 과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt
```

```
fig, loss_ax = plt.subplots()
```

```
acc_ax = loss_ax.twinx()
```

```
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
```

```
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
```

```
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
```

```
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
```

```
plt.show()
'''

'''
# 6. 모델 평가하기
loss_and_metrics = m.evaluate(X_test, y_test, batch_size=32)
```

```
print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))
```
