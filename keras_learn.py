# python -m pip install h5py
from __future__ import print_function
import numpy as np
import os

np.random.seed(1337)  # for reproducibility  用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随即数都相同

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

# 用于画图
import matplotlib.pyplot as pt
import keras
import matplotlib.pyplot as plt
'''
Olivetti Faces是纽约大学的一个比较小的人脸库，由40个人的400张图片构成，即每个人的人脸图片为10张。每张图片的灰度级为8位，每个像素的灰度大小位于0-255之间。整张图片大小是1190 × 942，一共有20 × 20张照片。那么每张照片的大小就是（1190 / 20）× （942 / 20）= 57 × 47 。
'''

# There are 40 different classes
nb_classes = 100  # 40个类别
epochs = 4  # 进行40轮次训
batch_size = 100  # 每次迭代训练使用40个样本

# input image dimensions
img_rows, img_cols = 100, 100
# number of convolutional filters to use
nb_filters1, nb_filters2 = 5, 10  # 卷积核的数目（即输出的维度）
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3  # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。

root = "./n22"


def getnum(x):
    # print(x)
    j = x.index('.')
    t = x[0:j]
    return int(t)


def list_all_Picfile(rootdir):
    _files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    return list


pathList = list_all_Picfile('./n22')
for i in range(len(pathList) - 1):
    for j in range(len(pathList) - 1 - i):
        if getnum(pathList[j]) > getnum(pathList[j + 1]):
            temp = pathList[j]
            pathList[j] = pathList[j + 1]
            pathList[j + 1] = temp


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def load_data2(dataset_path):
    faces = np.empty((1000, 10000 * 3))
    #list = os.listdir(dataset_path)

    row = 0
    col = 0
    times = 0
    for i in range(0, len(pathList)):
        path = os.path.join(dataset_path, pathList[i])
        img = Image.open(path)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        # print(img_ndarray.shape)
        # exit(0)
        faces[i] = np.ndarray.flatten(img_ndarray)



    label = np.empty(1000)
    for i in range(100):
        label[i * 10: i * 10 + 10] = i
    label = label.astype(np.int)

    # train:800,valid:100,test:100
    train_data = np.empty((800, 10000 * 3))
    train_label = np.empty(800)
    valid_data = np.empty((100, 10000 * 3))
    valid_label = np.empty(100)
    test_data = np.empty((100, 10000 * 3))
    test_label = np.empty(100)

    for i in range(100):
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]  # 训练集中的数据，取前8个
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]  # 训练集对应的标签
        valid_data[i] = faces[i * 10 + 8]  # 验证集中的数据
        valid_label[i] = label[i * 10 + 8]  # 验证集对应的标签
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')  # 之前取得长一点尽量避免py的存储误差

    rval = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]  # 组成元组返回
    return rval


def load_data(dataset_path):
    img = Image.open(dataset_path)

    img_ndarray = np.asarray(img, dtype='float64') / 255  # asarray，将数据转化为np.ndarray，但使用原内存
    # 400 pictures, size: 57*47 = 2679  除255是归一化，将所有值归到0~1之间
    faces = np.empty((400, 2679))  # 400行，一人脸一行，2679列因为像素这么多
    for row in range(20):  # 以人脸为一个单元坐标，原图20x20
        for column in range(20):
            faces[row * 20 + column] = np.ndarray.flatten(
                img_ndarray[row * 57: (row + 1) * 57, column * 47: (column + 1) * 47])
            # flatten将多维数组降为一维 一个人脸像素行开头到行末尾，列开头到列末尾

    label = np.empty(400)
    for i in range(40):
        label[i * 10: i * 10 + 10] = i
    label = label.astype(np.int)

    # train:320,valid:40,test:40
    train_data = np.empty((320, 2679))
    train_label = np.empty(320)
    valid_data = np.empty((40, 2679))
    valid_label = np.empty(40)
    test_data = np.empty((40, 2679))
    test_label = np.empty(40)

    for i in range(40):
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]  # 训练集中的数据，取前8个
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]  # 训练集对应的标签
        valid_data[i] = faces[i * 10 + 8]  # 验证集中的数据
        valid_label[i] = label[i * 10 + 8]  # 验证集对应的标签
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')  # 之前取得长一点尽量避免py的存储误差

    rval = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]  # 组成元组返回
    return rval


def set_model(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()  # 线性叠加模型
    if K.image_data_format() == 'channels_first':  # 只是统一格式，5个3x3的卷积核，输入1x100x100的图像
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape=(3, img_rows, img_cols)))
    else:
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape=(img_rows, img_cols, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, kernel_size=(3, 3)))  # 一般下一层的核数量都比上一层大
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000))  # Full connection 1000个神经元
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))  # 输出了
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  # 逐步梯度下降的参数
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    # 定义图形对象
    history = LossHistory()
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_val, Y_val),shuffle=True,callbacks=[history])
    # 绘制图像
    history.loss_plot('epoch')

    model.save_weights('model_weights.h5', overwrite=True)
    return model


def test_model(model, X, Y):
    model.load_weights('model_weights.h5')
    score = model.evaluate(X, Y, verbose=0)
    return score


if __name__ == '__main__':
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data2(r'./n22')

    if K.image_data_format() == 'channels_first':  # 在tensor里对数据类型判断 channel first是把通道数放在前面 1x 28 x 28
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_val = X_val.reshape(X_val.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:  # 这个是把通道放在后面 28 x 28 x 1
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)  # 1 为图像像素深度

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')  # 看数据集的大小
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = set_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    score = test_model(model, X_test, Y_test)

    model.load_weights('model_weights.h5')
    classes = model.predict_classes(X_test, verbose=0)
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("accuarcy:", test_accuracy)
    for i in range(0, 100):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classes[i]);
