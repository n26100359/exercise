# Mnist half imgae generator

Using mnist to train a the generator.

## Dataset

- Dataset: `mnist`

## Code Reference

- Google mnist_cnn.py: [mnist_cnn.py](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)

## Installation

`pip3 install -r requires.txt`

## Training pipeline

1. Train your model:
```
python3 adam_c.py
python3 adam_f.py
python3 SGD_c.py
python3 SGD_f.py
python3 batch_c.py
python3 batch_f.py
```

If you want to change the learning rate, total training steps or other training strategies, please modify the code in `adam_c.py`,`adam_f.py`,`SGD_c.py`,`SGD_f.py`,`batch_c.py`,`batch_f.py`.

## Tensorboard

Type the following command and check the url: http://localhost:6006

```
tensorboard --logdir==path/to/logs
```


## Run a simple chatbot

`adam_c.py`,`adam_f.py`,`SGD_c.py`,`SGD_f.py`,`batch_c.py`,`batch_f.py` 
will export a [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel)
Those models will be placed under the same folder.
