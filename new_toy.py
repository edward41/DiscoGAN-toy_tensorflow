__author__ = 'edward'
__author__ = 'edward'
from itertools import chain


from utils import *

from itertools import chain
import tensorflow as tf
from scipy.stats import norm
def get_feature_match_loss( feats_real, feats_fake):
        losses = []
        for real, fake in zip(feats_real, feats_fake):
            loss = tf.reduce_mean(tf.squared_difference(
                tf.reduce_mean(real, 0),
                tf.reduce_mean(fake, 0)),
                name='mse_feat_' + real.op.name)
            losses.append(loss)
        ret = tf.add_n(losses, name='feature_match_loss')
        return ret

'''
def generator(input, output, hidden_dims, name,reuse=False):
    hidden_dims=[128]*2
    with tf.variable_scope(name) as scope:
        if reuse:
		     scope.reuse_variables()
        prev=input
        h_list=[]

        for i, hidden_dim in enumerate(hidden_dims):
            h = tf.nn.relu(linear(prev, hidden_dim, 'g'+str(i)))
            h_list.append(h)
            prev = h
        return linear(prev, output,'g'+str(i+1))
'''
slim = tf.contrib.slim
def generator(input, name,reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()


# Get all the variables with the same given name, i.e. 'weights', 'biases'.
        h1 = tf.nn.relu(linear(input,128,scope='h1'))
        h2 = tf.nn.relu(linear(h1,128,scope='h2'))
        h3 = tf.nn.relu(linear(h2,128,scope='h3'))
        x = linear(h3,2,scope='x')
        '''
        h1 = tf.nn.dropout(tf.nn.relu(linear(input,128,scope='h1')),0.5)
        h2 = tf.nn.dropout(tf.nn.relu(linear(h1,128,scope='h2')),0.5)
        h3 = tf.nn.dropout(tf.nn.relu(linear(h2,128,scope='h3')),0.5)
        x = linear(h3,2,scope='x')

        h = input
        h = slim.repeat(h, 3, slim.fully_connected, 128, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, 2, activation_fn=None, scope="p_x")
        '''
        return x
def discriminator(input, name, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        h1 = tf.nn.dropout(tf.nn.relu(linear(input,128,scope='h1')),0.5)
        h2 = tf.nn.dropout(tf.nn.relu(linear(h1,128,scope='h2')),0.5)
        h3 = tf.nn.dropout(tf.nn.relu(linear(h2,128,scope='h3')),0.5)
        h4 = tf.nn.dropout(tf.nn.relu(linear(h3,128,scope='h4')),0.5)
        h5 = tf.nn.dropout(tf.nn.relu(linear(h4,128,scope='h5')),0.5)
        D_logit = linear(h5,1,scope='log_d')
        D_prob = tf.nn.sigmoid(D_logit)
        return D_logit, D_prob
        '''
        input = tf.concat(input, 1)
        h1 = tf.nn.relu(linear(input,128,scope='h1'))
        h2 = tf.nn.relu(linear(h1,128,scope='h2'))
        h3 = tf.nn.relu(linear(h2,128,scope='h3'))
        h4 = tf.nn.relu(linear(h3,128,scope='h4'))
        h5 = tf.nn.relu(linear(h4,128,scope='h5'))
        log_d = tf.sigmoid(linear(h5,1,scope='log_d'))
        return tf.squeeze(log_d, squeeze_dims=[1])

        h = tf.concat(input, 1)
        h = slim.repeat(h, 3, slim.fully_connected, 128, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=tf.sigmoid, scope="p_x")
        '''


'''
def discriminator(input, output, hidden_dims, name, reuse=False):
    hidden_dims=[128]*4
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        prev=input
        h_list=[]
        for i, hidden_dim in enumerate(hidden_dims):
            h = tf.nn.relu(linear(prev, hidden_dim, 'd'+str(i)))
            h_list.append(h)
            prev = h
        a =  tf.reshape(tf.sigmoid(linear(prev, output,'d'+str(i+1))),[-1,1])
        return a, h_list
'''
class GAN(object):
    def __init__(self, Flags) :
        #data, gen, num_steps, batch_size, minibatch, log_every
        self.Flags = Flags
        self.num_steps =  Flags.num_steps
        self.batch_size =  Flags.batch_size
        self.log_every =  Flags.log_every
        self.mlp_hidden_size = 128
        self.g_num_layer = 3
        self.d_num_layer = 5
        # can use a higher learning rate when not using the minibatch layer
        self.l_r = Flags.learning_rate
        self.beta1 = Flags.beta1
        self.beta2 = Flags.beta2
        self.L1_lambda = Flags.L1_lambda


        self._create_model()

    def _create_model(self):

        self.real_A = tf.placeholder(tf.float32,
                                        [self.batch_size, 2],
                                        name='real_A')
        #np.random.shuffle(data)
        self.real_B = tf.placeholder(tf.float32,
                                        [self.batch_size, 2],
                                        name='real_B')
        ''' generator & discirminator '''
        self.G_AB = generator(self.real_A ,name="g_AB")
        self.G_BA = generator(self.real_B,name="g_BA")
        self.G_ABA = generator(self.G_AB,name="g_BA",reuse=True)
        self.G_BAB = generator(self.G_BA,name="g_AB",reuse=True)
        
        self.real_AB = tf.concat([self.real_A, self.real_B], 1)
        self.fake_AB = tf.concat([self.real_A, self.G_AB], 1)
        self.real_BA = tf.concat([self.real_B, self.real_A], 1)
        self.fake_BA = tf.concat([self.real_B, self.G_BA], 1)
        
        with tf.variable_scope("cGAN_d"):
            self.AB_D, self.AB_D_logits =discriminator(self.real_AB,name= 'd_AB', reuse=False)
            self.AB_D_, self.AB_D_logits_ = discriminator(self.fake_AB,name= 'd_AB', reuse=True)
            self.BA_D, self.BA_D_logits =discriminator(self.real_BA,name= 'd_BA', reuse=False)
            self.BA_D_, self.BA_D_logits_ = discriminator(self.fake_BA,name= 'd_BA', reuse=True)
        with tf.variable_scope("DiscoGAN_d"):
            self.D_A_real = discriminator(self.real_A, name="d_A")
            self.D_A_fake = discriminator(self.G_BA, name="d_A",reuse=True)
    
            self.D_B_real = discriminator(self.real_B, name="d_B")
            self.D_B_fake = discriminator(self.G_AB, name="d_B",reuse=True)
        '''cGAN loss function '''

        self.d_loss_AB_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.AB_D_logits, labels=tf.ones_like(self.AB_D)))
        self.d_loss_AB_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.AB_D_logits_, labels=tf.zeros_like(self.AB_D_)))
        self.d_loss_BA_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.BA_D_logits, labels=tf.ones_like(self.BA_D)))
        self.d_loss_BA_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.BA_D_logits_, labels=tf.zeros_like(self.BA_D_)))
        self.g_loss_BA = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.BA_D_logits_, labels=tf.ones_like(self.BA_D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.G_AB))
        self.g_loss_AB = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.AB_D_logits_, labels=tf.ones_like(self.AB_D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_A - self.G_BA))
        self.Cg_loss = self.g_loss_AB + self.g_loss_BA
        self.Cd_loss = self.d_loss_AB_real + self.d_loss_AB_fake +  self.d_loss_BA_real + self.d_loss_BA_fake
        
        ''' DiscoGAN loss function '''
        l_const_a = tf.reduce_mean(tf.nn.l2_loss(self.G_ABA-self.real_A))
        l_const_b = tf.reduce_mean(tf.nn.l2_loss(self.G_BAB-self.real_B))

        l_gan_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_A_fake,labels=tf.ones_like(self.D_A_fake)))
        l_gan_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_B_fake,labels=tf.ones_like(self.D_B_fake)))

        #Real example loss for discriminators
        l_disc_a_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_A_real,labels=tf.ones_like(self.D_A_real)))
        l_disc_b_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_B_real,labels=tf.ones_like(self.D_B_real)))

        #Fake example loss for discriminators
        l_disc_a_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_A_fake,labels=tf.zeros_like(self.D_A_fake)))
        l_disc_b_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_B_fake,labels=tf.zeros_like(self.D_B_fake)))
        #Combined loss for individual discriminators
        #Combined loss for individual discriminators
        l_disc_a = l_disc_a_real +l_disc_a_fake
        l_disc_b = l_disc_b_real + l_disc_b_fake
        #fm_loss_A = get_feature_match_loss(self.D_A_feature_real, self.D_A_feature_fake)
        #fm_loss_B = get_feature_match_loss(self.D_B_feature_real, self.D_B_feature_fake)
        self.Dd_loss = l_disc_a +l_disc_b

        #Combined loss for individual generators
        l_ga = l_gan_a + l_const_a
        l_gb = l_gan_b + l_const_b
        self.Dg_loss= l_ga + l_gb #+ fm_loss_A +fm_loss_B
        
        self.Cd_params = []
        self.Dd_params = []
        self.g_params = []
        for v in tf.trainable_variables():
            if "cGAN_d" in v.name:
                self.Cd_params.append(v)
            if "DiscoGAN_d"in v.name:
                self.Dd_params.append(v)
            if 'g' in v.name:
                self.g_params.append(v)
        #self.fake_B_sample = self.sampler(self.real_A)

        #Combined loss for individual generators




        self.Dd_optim = tf.train.AdamOptimizer(self.l_r, beta1=self.beta1, beta2=self.beta2) \
                          .minimize(self.Dd_loss, var_list=self.Dd_params)
        self.Cd_optim = tf.train.AdamOptimizer(self.l_r, beta1=self.beta1, beta2=self.beta2) \
                          .minimize(self.Cd_loss, var_list=self.Cd_params)
        self.Dg_optim = tf.train.AdamOptimizer(self.l_r, beta1=self.beta1, beta2=self.beta2) \
                          .minimize(self.Dg_loss, var_list=self.g_params)
        self.Cg_optim = tf.train.AdamOptimizer(self.l_r, beta1=self.beta1, beta2=self.beta2) \
                          .minimize(self.Cg_loss, var_list=self.g_params)
        #self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        #self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        #print tf.variable_scope('d')
        #print self.d_params
        #print self.g_params
        #Discriminator for input b
        '''
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.l_r, global_step,
                                          2000, 0.8, staircase=True)

        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                          .minimize(self.l_disc, var_list=self.d_params,global_step=global_step)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                          .minimize(self.l_g, var_list=self.g_params,global_step=global_step)
        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        '''
    def train(self, data, sess):
        train_A, train_B = data
        #train_A[[0,1]] = train_A[[1,0]]
        #train_B[[0,1]]= train_B[[1,0]]
        #plt.plot(train_A, train_B)

        #

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        index = np.arange(len(train_A))

        np.random.shuffle(index)
        print index
        origin_train_A=train_A
        origin_train_B=train_B
        train_A, train_B = train_A[index], train_B[index]

        batch_idxs = min(len(train_A),len(train_B)) // self.batch_size
        id = 0
        for step in xrange(self.num_steps):
            #np.random.shuffle(train_A)
            #np.random.shuffle(train_B)
            '''
            for idx in xrange(0, batch_idxs):

                batch_files_A = train_A[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_files_B = train_B[idx*self.batch_size:(idx+1)*self.batch_size]

                # Update D network

                _,dLoss = sess.run([self.Dd_optim, self.Dd_loss],
                                              feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})
                _, gLoss = sess.run([self.Dg_optim, self.Dg_loss],
                                               feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})

                _,dLoss = sess.run([self.Cd_optim, self.Cd_loss],
                                              feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})
                _, gLoss = sess.run([self.Cg_optim, self.Cg_loss],
                                               feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})


                _,dLoss = sess.run([self.Dd_optim, self.Dd_loss],
                                              feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})
                _, gLoss = sess.run([self.Dg_optim, self.Dg_loss],
                                               feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})
            '''
            for idx in xrange(0, batch_idxs) :
                batch_files_A = train_A[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_files_B = train_B[idx*self.batch_size:(idx+1)*self.batch_size]

                # Update D network

                _,dLoss = sess.run([self.Dd_optim, self.Dd_loss],
                                              feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})
                _, gLoss = sess.run([self.Dg_optim, self.Dg_loss],
                                               feed_dict={ self.real_A: batch_files_A,
                                                           self.real_B: batch_files_B})
            print ("DiscoGAN, d Loss :{:.6}, g Loss : {:.6} ".format(dLoss,gLoss))

            for i in range(0,int(batch_idxs*0.05)) :
                if step % 10 == 0:
                    break;
                for idx in xrange(0, 1):
                    n = 2
                    size = self.batch_size/n
                    all_size = len(origin_train_A)/n
                    origin_batch_files_A = np.concatenate((origin_train_A[idx*size:(idx+1)*size],\
                                                           origin_train_A[all_size+idx*size:all_size+(idx+1)*size]), axis = 0)
                    origin_batch_files_B = np.concatenate((origin_train_B[idx*size:(idx+1)*size],\
                                                           origin_train_B[all_size+idx*size:all_size+(idx+1)*size]), axis = 0)
                    _,dLoss = sess.run([self.Cd_optim, self.Cd_loss],
                                                  feed_dict={ self.real_A: origin_batch_files_A,
                                                               self.real_B: origin_batch_files_B})
                    _, gLoss = sess.run([self.Cg_optim, self.Cg_loss],
                                                   feed_dict={ self.real_A: origin_batch_files_A,
                                                               self.real_B: origin_batch_files_B})
            print ("CGAN, d Loss :{:.6}, g Loss : {:.6} ".format(dLoss,gLoss))
            if ((step) % 10 == 0) :

                g_b,g_a,g_aba,g_bab = sess.run([self.G_AB,self.G_BA,self.G_ABA,self.G_BAB], feed_dict={self.real_A: batch_files_A,self.real_B: batch_files_B})
                print g_b.shape
                #g_aba = sess.run(self.G_ABA, feed_dict={self.real_A: batch_files_A})
                #g_bab = sess.run(self.G_BAB, feed_dict={self.real_B: batch_files_B})
                plot(batch_files_B, 'k.')
                plot(g_b, 'b.')
                plot(batch_files_A, 'k.')
                plot(g_a, 'r.')
                plot(g_aba, 'c.')
                plot(g_bab, 'm.')
                #plot(batch_files_A, 'k.')
                #plot2(g_a, 'b.')
                plt.show()
                print step


            #print ("d Loss :{:.6}, g Loss : {:.6} ".format(dLoss,gLoss))







                #print id
                #print sess.run(self.learning_rate)

                #next purpose : make animator


    def test(self, data,data_with_class,  sess):
        A_data_with_class, B_data_with_class = data_with_class
        A_data,B_data = data
        print A_data_with_class
        G_A = np.zeros(shape=(50000,2))
        G_B = np.zeros(shape=(50000,2))
        i = 0
        for key, value in B_data_with_class.items():
            print i
            for minibatch in xrange(0,len(value) // self.batch_size) :
                pred = sess.run(self.G_BA, feed_dict={self.real_B:value[self.batch_size*minibatch:self.batch_size*(minibatch+1)]})
                G_A[i*self.batch_size:(i+1)*self.batch_size] = pred
            i+=1
        i = 0
        for key, value in A_data_with_class.items():
            for minibatch in xrange(0,len(value) // self.batch_size) :
                pred = sess.run(self.G_AB, feed_dict={self.real_A:value[self.batch_size*minibatch:self.batch_size*(minibatch+1)]})
                G_B[i*self.batch_size:(i+1)*self.batch_size] = pred
            i+=1
        plot(A_data, '.')
        plot(B_data, '.')
        plot(G_A, 'r.')
        plot(G_B, 'b.')
        plt.show()




