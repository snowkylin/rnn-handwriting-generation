import tensorflow as tf
import numpy as np
import argparse
import model as m

from utils import DataLoader

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
# parser.add_argument('--decay_rate', type=float, default=0.95,
#                      help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=0.8,
                    help='dropout keep probability')
# parser.add_argument('--grad_clip', type=float, default=10.,
#                      help='clip gradients at this value')
args = parser.parse_args()

data_loader = DataLoader(args.batch_size, args.T, args.data_scale)
model = m.Model(args)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    for e in range(args.num_epochs):
        print "epoch %d" % e
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            x, y, c = data_loader.next_batch()
            print c
            # print x
            # print sess.run([model.loss_gaussian, model.loss_bernoulli],
            #                feed_dict={model.x: x, model.y: y})

            sess.run(model.train_op, feed_dict={model.x: x, model.y: y})
            if b % 100 == 0:
                loss = sess.run(model.loss, feed_dict={model.x: x, model.y: y})
                print 'batches %d, loss %g' % (b, loss)
        saver.save(sess, 'save/model.tfmodel', global_step=e)
