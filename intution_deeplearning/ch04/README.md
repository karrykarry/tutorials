#Usage
-Training:
```
python dcgan.py --mode train
```  

-Generation:
```
python dcgan.py --mode generate
```

#MNISTを偽造するための敵対的なGAN
+ environment : anaconda

```
    ~$cd
    ~$git clone  https://github.com/bstriner/keras-adversarial
    ~$cd keras-adversarial
```
+ fix code(https://github.com/bstriner/keras-adversarial/pull/59/files)
```
    ~$vim adversarial_model.py
    ~$cd ../
    ~$python setup.py install
    ~$cd examples
    ~$cp image_utils.py ~/tutorials/intution_deeplearning/ch04/
    ~$cd ~/tutorials/intution_deeplearning/ch04/
    ~$python example_gan_convolutional.py
```



