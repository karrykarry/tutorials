#cifar10で学習した重みを使って学習判定している

from pathlib import Path
import pathlib
import numpy as np
from PIL import Image
from keras.models import load_model

from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

from keras.utils import plot_model

print("0-airplane")
print("1-automobile")
print("2-bird")
print("3-cat")
print("4-deer")
print("5-dog")
print("6-frog")
print("7-horse")
print("8-ship")
print("9-truck")


# load model
model_path = "logdir_cifar10_deep_with_aug/model_file.hdf5"
model = load_model(model_path)

# images_folder = "sample_images"
home_folder = "/home/amsl/Pictures"
images_folder = "/cifar10/"
path = home_folder + images_folder 

# image shape
image_shape = (32, 32, 1)


#load images
def crop_resize(image_path):
    image = Image.open(image_path)
    length = min(image.size)
    crop = image.crop((0, 0,length ,length))
    resized = crop.resize(image_shape[:2])
    # resized = crop.resize(32, 32, 1)
    img = np.array(resized).astype("float32")
    img /= 255
    return img


def ref():
    folder = Path(images_folder)
    image_paths = [str(f) for f in folder.glob("*.png")]
    print("image:" + str(image_paths[0]))
    images = [crop_resize(p) for p in image_paths]
    images = np.array(images)
    print(images)
    predicted = model.predict_classes(images)
    print(predicted[0])


def show_image(image, predict_r):
    plt.imshow(image)
    plt.title('pred:{}'.format(predict_r))
    plt.show()


# def ref2():
##画像読み込み(配列で読み込みたかった)
folder = Path(path)
image_paths = [str(f) for f in folder.glob("*")]

# print("result" + str(f) for f in image_paths)
print(image_paths)
print("aaaaaaaaaaaaaa")
# temp_img=load_img((str(f) for f in image_paths) ,target_size=(32,32))
# temp_img=load_img(path + "/dog.jpeg",target_size=(32,32))
temp_img=load_img(path + "/yamcha.jpg",target_size=(32,32))

##画像を配列に変換し0-1で正規化
temp_img_array=img_to_array(temp_img)
temp_img_array=temp_img_array.astype('float32')/255.0
temp_img_array=temp_img_array.reshape((1,32,32,3))

##モデルを表示
## モデルの可視化もやりたい
model.summary()

##画像を予想
img_pred=model.predict_classes(temp_img_array)
print('\npredict_classes=',img_pred)

##画像を表示
show_image(temp_img, img_pred)

