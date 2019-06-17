###重みの共有

model = Sequential()
#256x256 ピクセルのRGBの画像=(256, 256, 3)を3x3のフィルタ32個で畳み込む場合
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))

#フィルタが正方形である場合、kernel_sizeを指定する。
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(256, 256, 3)))
