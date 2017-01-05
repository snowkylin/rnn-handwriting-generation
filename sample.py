import tensorflow as tf
import numpy as np
import model as m
from utils import *
from config import *

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
str = 'a quick brown fox jumps over the lazy dog'
# str = 'aaaaabbbbbccccc'
args.U = len(str)
args.c_dimension = len(data_loader.chars) + 1
args.T = 1
args.batch_size = 1
args.action = 'sample'

model = m.Model(args)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('save_%s' % args.mode)

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    if args.mode == 'predict':
        strokes = model.sample(sess, 800)
    if args.mode == 'synthesis':
        str_vec = vectorization(str, data_loader.char_to_indices)
        strokes = model.sample(sess, len(str) * args.points_per_char, str=str_vec)
    # print strokes
    draw_strokes_random_color(strokes, factor=0.1, svg_filename='sample' + '.normal.svg')
