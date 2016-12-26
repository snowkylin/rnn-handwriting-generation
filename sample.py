import tensorflow as tf
import numpy as np
import argparse
import model as m

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='synthesis',
                    help='predict or synthesis')
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
parser.add_argument('--chars', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
                    help='chars')
parser.add_argument('--num_epochs', type=int, default=30,
                    help='num of epochs')
parser.add_argument('--T', type=int, default=300,
                    help='RNN sequence length')
parser.add_argument('--points_per_char', type=int, default=25,
                    help='points per char (appr.)')
parser.add_argument('--rnn_state_size', type=int, default=256,
                    help='RNN hidden state size')
parser.add_argument('--num_layers', type=int, default=2,
                    help='num of RNN stack layers')
parser.add_argument('--M', type=int, default=20,
                    help='num of mixture bivariate gaussian')
parser.add_argument('--K', type=int, default=10,
                    help='num of mixture bivariate gaussian (for synthesis)')
parser.add_argument('--data_scale', type=float, default=20,
                    help='factor to scale raw data down by')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
# parser.add_argument('--decay_rate', type=float, default=0.95,
#                      help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=0.8,
                    help='dropout keep probability')
# parser.add_argument('--grad_clip', type=float, default=10.,
#                      help='clip gradients at this value')
args = parser.parse_args()
args.T = 1
args.batch_size = 1

data_loader = DataLoader(50, 300, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
str = 'a quick brown fox jumps over the lazy dog'
args.U = len(str)
args.c_dimension = len(data_loader.chars) + 1

model = m.Model(args)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('save_%s' % args.mode)

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    if args.mode == 'predict':
        strokes = model.sample(sess, 800)
    if args.mode == 'synthesis':
        str = vectorization(str, data_loader.char_to_indices)
        strokes = model.sample(sess, 800, str=str)
    # print strokes
    draw_strokes_random_color(strokes, factor=0.1, svg_filename='sample' + '.normal.svg')