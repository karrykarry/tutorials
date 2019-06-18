#最初期のCNNであり、単純な画像の変形や歪みに頑健というCNNの基本的な特性をMNISTの画像認識タスクにおいて証明したのがLeNet
import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import RMSprop
# from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

####### ネットワーク ######

def network(input_shape, num_classes):
    model = Sequential()
#extract image features by convolution and max pooling layers
    model.add(Conv2D(32, kernel_size=5, padding="same",
        input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) #処理後の特徴マップは14x14x20
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding="same",
        activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) #処理後の特徴マップは7x7x20
# classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax")) #ソフトマックス関数を適用することで値を0~1の確率値に変更します。
    return model

##########################

####### 学習させるためのデータセットを用意します。 ########
# 今回はMNISTの手書き数字を用意します。

class CIFAR10Dataset():

    def __init__(self):
        self.image_shape = (32, 32, 3) # image is 28x28x1 (grayscale)
        self.num_classes = 10

    def get_batch(self):
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test

# 画像データはpreprocessにより0~1の値に変換し。正解データ(ラベルデータ)はone-hotベクトルに変換します。
#
    def preprocess(self, data, label_data=False):
    
        if label_data:
#convert class vectors to binary class matrices
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255 # convert the value to 0~1 scale
            shape = (data.shape[0],) + self.image_shape # add dataset length to top
            data = data.reshape(shape)
            
        return data

#############################################################

# ネットワークを学習させる Trainer を実装します。
#学習させるmodel、最小化すべき誤差であるloss、最適の方法であるoptimizerを受け取り、trainで学習データを受け取り学習させます。

class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), "logdir")
        self.model_file_name = "model_file.hdf5"

    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir) # remove previous execution
            os.mkdir(self.log_dir)

        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(os.path.join(self.log_dir, self.model_file_name),
                    save_best_only=True)
                ],
            verbose=self.verbose
        )


#############################################

####### 最後に定義した処理を呼び出し実際に学習を行う処理を実装します。


# dataset = MNISTDataset()
dataset = CIFAR10Dataset()


# make model
# model = lenet(dataset.image_shape, data.num_classes)
model = network(dataset.image_shape, data.num_classes)

#train the model
x_train, y_train, x_test, y_test = dataset.get_batch()

# trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam())
trainer = Trainer(model, loss="categorical_crossentropy", optimizer=RMSprop())

trainer.train(x_train, y_train, batch_size=128, epochs=12, validation_split=0.2)

#show result
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])



