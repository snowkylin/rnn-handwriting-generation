import tensorflow as tf
import numpy as np
from utils import vectorization

class Model():
    def __init__(self, args):
        def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            z = tf.square((x1 - mu1) / sigma1) + tf.square((x2 - mu2) / sigma2) \
                - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            return tf.exp(-z / (2 * (1 - tf.square(rho)))) / \
                   (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho)))

        def expand(x, dim, N):
            return tf.concat(dim, [tf.expand_dims(x, dim) for _ in range(N)])

        self.args = args

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, args.T, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, args.T, 3])

        x = tf.split(1, args.T, self.x)
        x_list = [tf.squeeze(x_i, [1]) for x_i in x]
        if args.mode == 'predict':
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size)
            self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * args.num_layers)
            # if (args.keep_prob < 1):  # training mode
            #     self.stacked_cell = tf.nn.rnn_cell.DropoutWrapper(self.stacked_cell, output_keep_prob=args.keep_prob)
            self.init_state = self.stacked_cell.zero_state(args.batch_size, tf.float32)

            self.output_list, self.final_state = tf.nn.rnn(self.stacked_cell, x_list, self.init_state)
            # self.output_list, self.final_state = tf.nn.seq2seq.rnn_decoder(x_list, self.init_state, self.stacked_cell)
        if args.mode == 'synthesis':
            self.c_vec = tf.placeholder(dtype=tf.float32, shape=[None, args.U, args.c_dimension])
            self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size)
            self.cell2 = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_state_size)
            self.init_cell1_state = self.cell1.zero_state(args.batch_size, tf.float32)
            self.init_cell2_state = self.cell2.zero_state(args.batch_size, tf.float32)
            cell1_state = self.init_cell1_state
            cell2_state = self.init_cell2_state
            self.output_list = []
            h2k_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.rnn_state_size, args.K * 3]))
            h2k_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.K * 3]))
            kappa_prev = tf.zeros([args.batch_size, args.K, args.U])
            w = tf.zeros([args.batch_size, args.c_dimension])
            u = expand(expand(np.array([i for i in range(args.U)], dtype=np.float32), 0, args.K), 0, args.batch_size)
            DO_SHARE = False
            for t in range(args.T):
                with tf.variable_scope("cell1", reuse=DO_SHARE):
                    h_cell1, cell1_state = self.cell1(tf.concat(1, [x_list[t], w]), cell1_state)
                k_gaussian = tf.nn.xw_plus_b(tf.reshape(h_cell1, [-1, args.rnn_state_size]),
                                    h2k_w, h2k_b)
                alpha_hat, beta_hat, kappa_hat = tf.split(1, 3, k_gaussian)
                alpha = tf.expand_dims(tf.exp(alpha_hat), 2)
                beta = tf.expand_dims(tf.exp(beta_hat), 2)
                kappa = kappa_prev + tf.expand_dims(tf.exp(kappa_hat), 2)
                kappa_prev = kappa
                phi = tf.reduce_sum(tf.exp(tf.square(-u + kappa) * (-beta)) * alpha, 1)
                w_list = [0] * args.batch_size
                for batch in range(args.batch_size):
                    w_list[batch] = tf.matmul(phi[batch: batch + 1, :], self.c_vec[batch, :, :])
                w = tf.concat(0, w_list)
                with tf.variable_scope("cell2", reuse=DO_SHARE):
                    output_t, cell2_state = self.cell2(
                        tf.concat(1, [x_list[t], h_cell1, w]), cell2_state)
                # with tf.variable_scope("cell1", reuse=DO_SHARE):
                #     output_t, cell1_state = self.cell1(x_list[t], cell1_state)
                self.output_list.append(output_t)
                DO_SHARE = True
            self.final_cell1_state = cell1_state
            self.final_cell2_state = cell2_state

        NOUT = 1 + args.M * 6  # end_of_stroke, num_of_gaussian * (pi + 2 * (mu + sigma) + rho)
        output_w = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[args.rnn_state_size, NOUT]))
        output_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[NOUT]))

        self.output = tf.nn.xw_plus_b(tf.reshape(tf.concat(1, self.output_list), [-1, args.rnn_state_size]),
                                      output_w, output_b)
        y1, y2, y_end_of_stroke = tf.unpack(tf.reshape(self.y, [-1, 3]), axis=1)

        self.end_of_stroke = 1 / (1 + tf.exp(self.output[:, 0]))
        pi_hat, self.mu1, self.mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(1, 6, self.output[:, 1:])
        pi_exp = tf.exp(pi_hat)
        pi_exp_sum = tf.reduce_sum(pi_exp, 1)
        self.pi = pi_exp / expand(pi_exp_sum, 1, args.M)
        self.sigma1 = tf.exp(sigma1_hat)
        self.sigma2 = tf.exp(sigma2_hat)
        self.rho = tf.tanh(rho_hat)
        self.gaussian = self.pi * bivariate_gaussian(
            expand(y1, 1, args.M), expand(y2, 1, args.M),
            self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho
        )
        eps = 1e-20
        self.loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(self.gaussian, 1) + eps))
        self.loss_bernoulli = tf.reduce_sum(
            -tf.log((self.end_of_stroke + eps) * y_end_of_stroke
                    + (1 - self.end_of_stroke + eps) * (1 - y_end_of_stroke))
        )

        self.loss = (self.loss_gaussian + self.loss_bernoulli) / (args.batch_size * args.T)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def sample(self, sess, length, str=None):
        x = np.zeros([1, 1, 3], np.float32)
        x[0, 0, 2] = 1
        strokes = np.zeros([length, 3], dtype=np.float32)
        strokes[0, :] = x[0, 0, :]
        if self.args.mode == 'predict':
            state = sess.run(self.stacked_cell.zero_state(1, tf.float32))
        if self.args.mode == 'synthesis':
            cell1_state = sess.run(self.cell1.zero_state(1, tf.float32))
            cell2_state = sess.run(self.cell2.zero_state(1, tf.float32))
        for i in range(length - 1):
            if self.args.mode == 'predict':
                feed_dict = {self.x: x, self.init_state: state}
                end_of_stroke, pi, mu1, mu2, sigma1, sigma2, rho, state = sess.run(
                    [self.end_of_stroke, self.pi, self.mu1, self.mu2,
                     self.sigma1, self.sigma2, self.rho, self.final_state],
                    feed_dict=feed_dict
                )
            if self.args.mode == 'synthesis':
                feed_dict = {self.x: x,
                             self.c_vec: str,
                             self.init_cell1_state: cell1_state,
                             self.init_cell2_state: cell2_state}
                end_of_stroke, pi, mu1, mu2, sigma1, sigma2, rho, cell1_state, cell2_state = sess.run(
                    [self.end_of_stroke, self.pi, self.mu1, self.mu2,
                     self.sigma1, self.sigma2, self.rho, self.final_cell1_state, self.final_cell2_state],
                    feed_dict=feed_dict
                )
            x = np.zeros([1, 1, 3], np.float32)
            r = np.random.rand()
            accu = 0
            for m in range(self.args.M):
                accu += pi[0, m]
                if accu > r:
                    x[0, 0, 0:2] = np.random.multivariate_normal(
                        [mu1[0, m], mu2[0, m]],
                        [[np.square(sigma1[0, m]), rho[0, m] * sigma1[0, m] * sigma2[0, m]],
                         [rho[0, m] * sigma1[0, m] * sigma2[0, m], np.square(sigma2[0, m])]]
                    )
                    break
            e = np.random.rand()
            if e < end_of_stroke:
                x[0, 0, 2] = 1
            else:
                x[0, 0, 2] = 0
            strokes[i + 1, :] = x[0, 0, :]
        return strokes