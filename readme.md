## Generating handwriting with LSTM, Mixture Gaussian & Bernoulli distribution and TensorFlow

This is a TensorFlow implementation of *[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)* by Alex Graves.

It have two functions:

* **Handwriting Prediction**: Randomly generate a line of handwriting (set `mode=predict`). 
* **Handwriting Synthesis**: Given a string, generate the corresponding handwriting (set `mode=synthesis`).

This project is adapted from [hardmaru's great work](https://github.com/hardmaru/write-rnn-tensorflow). The `util.py` is just from there and before running `train.py` and `sample.py` you need to follow [the instruction](https://github.com/hardmaru/write-rnn-tensorflow#training) and download the necessary files.

I hope to make this model simple, just **show the main algorithm as clear as possible** without struggling with bundles of optimization methods in deep learning. But if you wish, you can add them easily by yourself.

### Sample Result

#### Handwriting Prediction

![sample.normal.svg](https://cdn.rawgit.com/snowkylin/rnn-handwriting-generation/master/sample.normal.svg)
This is the result with default setting:
* rnn state size = 256
* rnn length = 300
* num of layers = 2
* number of mixture gaussian = 20

and 20+ epochs. Not so fancy but can be recognized as something like handwritting, huh?

#### Handwriting Synthesis

![sample.normal.biased.3.svg](https://cdn.rawgit.com/snowkylin/rnn-handwriting-generation/master/sample.normal.biased.3.svg)

This is the result with the string "a quick brown fox jumps over the lazy dog".

In addition, [the scribe project by greydanus](https://github.com/greydanus/scribe) also helps me a lot, expecially the use of `tf.batch_matmul()`.
