import tensorflow as tf
import numpy as np
import argparse
import model as m

from utils import DataLoader

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

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1
model = m.Model(args)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    for e in range(args.num_epochs):
        print "epoch %d" % e
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            x, y, c = data_loader.next_batch()
            if args.mode == 'predict':
                feed_dict = {model.x: x, model.y: y}
            if args.mode == 'synthesis':
                feed_dict = {model.x: x, model.y: y, model.c_vec: c}
            # print c
            # print x
            # print sess.run([model.phi],
            #                feed_dict=feed_dict)
            if b % 100 == 0:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print 'batches %d, loss %g' % (b, loss)
            sess.run(model.train_op, feed_dict=feed_dict)
        saver.save(sess, 'save_%s/model.tfmodel' % args.mode, global_step=e)
