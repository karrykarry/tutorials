#$B:G=i4|$N(BCNN$B$G$"$j!"C1=c$J2hA|$NJQ7A$dOD$_$K4h7r$H$$$&(BCNN$B$N4pK\E*$JFC@-$r(BMNIST$B$N2hA|G'<1%?%9%/$K$*$$$F>ZL@$7$?$N$,(BLeNet
import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

####### $B%M%C%H%o!<%/(B ######

del lenet(input_shape, num_classes):
    model = Sequential()
#extract image features by convolution and max pooling layers
    model.add(Conv2D(20, kernel_size=5, padding="same",
        input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) #$B=hM}8e$NFCD'%^%C%W$O(B14x14x20
    model.add(Conv2D(50, kernel_size=5, padding="same",
        activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) #$B=hM}8e$NFCD'%^%C%W$O(B7x7x20
# classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax")) #$B%=%U%H%^%C%/%94X?t$rE,MQ$9$k$3$H$GCM$r(B0~1$B$N3NN(CM$KJQ99$7$^$9!#(B
    return model

##########################

####### $B3X=,$5$;$k$?$a$N%G!<%?%;%C%H$rMQ0U$7$^$9!#(B ########
# $B:#2s$O(BMNIST$B$N<j=q$-?t;z$rMQ0U$7$^$9!#(B

class MNISTDataset():

    def __init__(self):
        self.image_shape = (28, 28, 1) # image is 28x28x1 (grayscale)
        self.num_classes = 10

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test

# $B2hA|%G!<%?$O(Bpreprocess$B$K$h$j(B0~1$B$NCM$KJQ49$7!#@52r%G!<%?(B($B%i%Y%k%G!<%?(B)$B$O(Bone-hot$B%Y%/%H%k$KJQ49$7$^$9!#(B
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

# $B%M%C%H%o!<%/$r3X=,$5$;$k(B Trainer $B$r<BAu$7$^$9!#(B
#$B3X=,$5$;$k(Bmodel$B!":G>.2=$9$Y$-8m:9$G$"$k(Bloss$B!":GE,$NJ}K!$G$"$k(Boptimizer$B$r<u$1<h$j!"(Btrain$B$G3X=,%G!<%?$r<u$1<h$j3X=,$5$;$^$9!#(B

class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), "logdir")

    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir) # remove previous execution

            os.mkdir(self.log_dir)

        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            callbacks=[TensorBoard(log_dir=self.log_dir)],
            verbose=self.verbose
        )


#############################################

####### $B:G8e$KDj5A$7$?=hM}$r8F$S=P$7<B:]$K3X=,$r9T$&=hM}$r<BAu$7$^$9!#(B


dataset = MNISTDataset()


# make model
model = lenet(dataset.image_shape, data.num_classes)

#train the model



































