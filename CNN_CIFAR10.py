"""
卷积神经网络(CNN)代码：在CIFAR-10上
"""

# 导入数据
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# 首次使用时会在线进行数据集下载
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('图像数据格式: ', X_train.shape)
print("训练集：%2.0f，测试集：%2.0f" %(X_train.shape[0], X_test.shape[0]))

X_train[0][:5, :, 1] # 图片矩阵中的第一条记录，显示R矩阵前5行

fig, ax = plt.subplots()
ax.imshow(X_train[1])

fig = plt.figure(figsize = (20, 5))
for i in range(20):
    ax = fig.add_subplot(2, 10, i + 1, xticks = [], yticks = [])
    ax.imshow(X_train[i])
y_train[:20].reshape(2, 10)

# 对因变量做处理
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import *
from keras.backend import *

model = Sequential()
# 设定卷积层，共32个卷积核，感受野3*3，不补零，relu连接
model.add(Conv2D(32, (3, 3), padding = 'valid', input_shape = (32, 32, 3),
                 activation = 'relu'))
# 设定池化层为2*2取最大值
model.add(MaxPooling2D(pool_size = (2, 2)))
# 将数据展平为普通的一维格式
model.add(Flatten())
# 设定一个普通的全连接层
model.add(Dense(128, activation = 'relu'))
# 设定输出层
model.add(Dense(10, activation = 'softmax'))

# 卷积层连接权重数：((3 * 3) * 3 + 1) * 32
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10)
model.evaluate(X_test, y_test)

