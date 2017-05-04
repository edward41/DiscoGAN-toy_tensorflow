from __future__ import division
from six.moves import xrange
import seaborn as sns
from util import *
sns.set(style="white")
random_seed = 123

class DiscoGAN(object):
    def __init__(self, sess, args):

        self.args = args
        self.sess = sess
        self.batch_size = args.batch_size
        self.hidden_layer_size = 128
        self.g_layer = 3
        self.d_layer = 5
        self.build_model()


    def build_model(self):
        self.real_A = tf.placeholder(tf.float32,
                                     [self.batch_size, 2],
                                     name='real_A')

        self.real_B = tf.placeholder(tf.float32,
                                     [self.batch_size, 2],
                                     name='real_B')

        self.G_AB = self.generator(self.real_A, name="gen_AB")
        self.G_BA = self.generator(self.real_B, name="gen_BA")

        self.G_ABA = self.generator(self.G_AB, name="gen_BA", reuse=True)
        self.G_BAB = self.generator(self.G_BA, name="gen_AB", reuse=True)

        disc_a_real = self.discriminator(self.real_A, name="d_a")
        disc_a_fake = self.discriminator(self.G_BA, name="d_a", reuse=True)

        disc_b_real = self.discriminator(self.real_B, name="d_b")
        disc_b_fake = self.discriminator(self.G_AB, name="d_b", reuse=True)

        '''
        Pytorch : nn.MSELoss = (self.G_ABA-self.real_A)^2*1/n(n : instance)
        '''
        l_const_a = tf.reduce_mean(tf.square(self.G_ABA-self.real_A))
        l_const_b = tf.reduce_mean(tf.square(self.G_BAB-self.real_B))

        '''
        Pytorch : nn.BCELoss(binary Cross Entropy)
        If you are using tensorflow, then can use sigmoid_cross_entropy_with_logits. 
        '''

        self.l_gan_a = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_a_fake, labels=tf.ones_like(disc_a_fake)))
        self.l_gan_b = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_b_fake, labels=tf.ones_like(disc_b_fake)))

        '''
        Pytorch : 
        l_gan_A = bce(D_A(x_BA), real_tensor)
        l_gan_B = bce(D_B(x_AB), real_tensor)
        '''
        l_disc_a_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_a_real, labels=tf.ones_like(disc_a_real)))
        l_disc_b_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_b_real, labels=tf.ones_like(disc_b_real)))

        l_disc_a_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_a_fake, labels=tf.zeros_like(disc_a_fake)))
        l_disc_b_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_b_fake, labels=tf.zeros_like(disc_b_fake)))

        '''
        Pytorch : 
        l_d_A_real, l_d_A_fake = bce(D_A(x_A), real_tensor), bce(D_A(x_BA), fake_tensor)
        l_d_B_real, l_d_B_fake = bce(D_B(x_B), real_tensor), bce(D_B(x_AB), fake_tensor)
        '''
        l_disc_a = l_disc_a_real + l_disc_a_fake
        l_disc_b = l_disc_b_real + l_disc_b_fake

        self.l_disc = l_disc_a + l_disc_b

        l_ga = self.l_gan_a + l_const_a
        l_gb = self.l_gan_b + l_const_b
        self.l_g = l_ga + l_gb

        # Parameter lists
        self.disc_params = []
        self.gen_params = []
        for v in tf.trainable_variables():
            if 'd' in v.name:
                self.disc_params.append(v)
            if 'gen' in v.name:
                self.gen_params.append(v)

        self.saver = tf.train.Saver()


    def train(self, args, train_data):
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.l_disc, var_list=self.disc_params)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1, beta2=args.beta2) \
            .minimize(self.l_g, var_list=self.gen_params)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        data_A, data_B = train_data

        id = 0
        np.random.shuffle(data_A)
        np.random.shuffle(data_B)
        for epoch in xrange(args.epoch):
            batch_idxs = min(len(data_A), len(data_B)) // self.batch_size

            for idx in xrange(batch_idxs):

                batch_images_A = data_A[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images_B = data_B[idx * self.batch_size:(idx + 1) * self.batch_size]

                # Update D network
                _, dLoss = self.sess.run([d_optim, self.l_disc],
                                         feed_dict={self.real_A: batch_images_A,
                                                    self.real_B: batch_images_B})

                # Update G network
                _, gLoss = self.sess.run([g_optim, self.l_g],
                                         feed_dict={self.real_A: batch_images_A,
                                                    self.real_B: batch_images_B})
                if (id % args.save_iteration_freq == 1) :
                    self.save(args.checkpoint_dir,id)
                if (id % args.check_results_freq == 0) :
                    print("Epoch: [%2d] [%4d/%4d] "
                          "\n\tDiscriminator Loss: %.8f, Generator Loss: %.8f" \
                    % (epoch, idx, batch_idxs,dLoss,gLoss))
                    fake_b = self.sess.run(self.G_AB, feed_dict={self.real_A: batch_images_A})
                    fake_a = self.sess.run(self.G_BA, feed_dict={self.real_B: batch_images_B})

                    plt.clf()

                    plot(batch_images_A,'k.')
                    plot(batch_images_B,'k.')
                    plot(fake_a,'.')
                    plot(fake_b,'.')

                    plt.savefig(os.path.join(args.result_dir, str(id)))
                id += 1

    def discriminator(self, input, name="discriminator", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            h=input
            for layer in xrange(self.d_layer) :
                next_h = tf.nn.relu(tf.contrib.layers.batch_norm(linear(h,self.hidden_layer_size,name='d_h'+str(layer))))
                h = next_h
            '''
            # batch normalization : deals with poor initialization helps gradient flow
            h1 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(input, self.hidden_layer_size, name='d_h1')))
            h2 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(h1, self.hidden_layer_size, name='d_h2')))
            h3 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(h2, self.hidden_layer_size, name='d_h3')))
            h4 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(h3, self.hidden_layer_size, name='d_h4')))
            h5 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(h4, self.hidden_layer_size, name='d_h5')))
            '''
            return tf.nn.sigmoid(linear(h,1), name='d_out')

    def generator(self, input, name="generator", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            h=input
            for layer in xrange(self.g_layer) :
                next_h = tf.nn.relu(linear(h,self.hidden_layer_size,name='g_h'+str(layer)))
                h = next_h
            '''
            h1 = tf.nn.relu(linear(h1, self.hidden_layer_size, name='g_h1'))
            h2 = tf.nn.relu(linear(h1, self.hidden_layer_size, name='g_h2'))
            h3 = tf.nn.relu(linear(h2, self.hidden_layer_size, name='g_h3'))
            '''
            return linear(h,2, name='g_out')

    def save(self, checkpoint_dir, step):
        model_name = "Disco_toy.model".format(step)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(" [*] Load {} file check point ").format(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Complete load checkpoint ")
            return True

        else:
            print(" [*] Fail load checkpoint ")
            return False

    def test(self, args, test_data):
        A_data_with_class, B_data_with_class = test_data
        for key, value in A_data_with_class.items():
            plot(value, 'k.')
        for key, value in B_data_with_class.items():
            plot(value, 'k.')
        batch_idxs = len(value) // self.batch_size
        for key, value in A_data_with_class.items():
            for idx in xrange(batch_idxs) :
                batch_A=value[idx*self.batch_size:(idx+1)*self.batch_size]
                fake_b = self.sess.run(self.G_AB, feed_dict={self.real_A: batch_A})
                plot(fake_b, 'b.')
        for key, value in B_data_with_class.items():
            for idx in xrange(batch_idxs) :
                batch_B =value[idx*self.batch_size:(idx+1)*self.batch_size]
                fake_a = self.sess.run(self.G_BA, feed_dict={self.real_B: batch_B})
                plot(fake_a, 'r.')



        plt.savefig(os.path.join(args.result_dir, 'test_result'))

