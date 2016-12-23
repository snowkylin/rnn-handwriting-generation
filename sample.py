import tensorflow as tf
import numpy as np
import argparse
import model as m

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=30,
                    help='num of epochs')
parser.add_argument('--T', type=int, default=300,
                    help='RNN sequence length')
parser.add_argument('--rnn_state_size', type=int, default=256,
                    help='RNN hidden state size')
parser.add_argument('--num_layers', type=int, default=2,
                    help='num of RNN stack layers')
parser.add_argument('--M', type=int, default=20,
                    help='num of mixture bivariate gaussian')
parser.add_argument('--data_scale', type=float, default=20,
                    help='factor to scale raw data down by')
parser.add_argument('--learning_rate', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=0.8,
                    help='dropout keep probability')
parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
args = parser.parse_args()
args.T = 1
args.batch_size = 1

model = m.Model(args)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('save')

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    strokes = model.sample(sess, 1000)
    # print strokes
    draw_strokes_random_color(strokes, factor=0.1, svg_filename='sample' + '.normal.svg')