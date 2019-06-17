from keras.model import Sequential
model = Sequential()
#入力が8次元、出力が12次元の層を定義
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))
