from pathlib import Path
import pathlib
import numpy as np
from PIL import Image
from keras.models import load_model

model_path = "logdir_cifar10_deep_with_aug/model_file.hdf5"
images_folder = "sample_images"

# images_folder = "~/Pictures/cifar10"

#load model
model = load_model(model_path)
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



folder = Path(images_folder)
print(folder)
image_paths = [str(f) for f in folder.glob("*.png")]
print("image:" + str(image_paths[0]))
images = [crop_resize(p) for p in image_paths]
images = np.array(images)
print(images)
predicted = model.predict_classes(images)

# print("result:" + str(predicted[0]))


image_paths2 = str('sample_images2/dog.png') 
print("image2:" + str(image_paths2))
images2 = crop_resize(image_paths2)
images2 = np.array(images2)
print(images2)
# print("path:" + str(image_paths2))
predicted2 = model.predict_classes(images2)


# assert predicted[0] == 3,"image should be cat"
# assert predicted[1] == 5,"image should be dog"

print("You can detect cat & dog")
