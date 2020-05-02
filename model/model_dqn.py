import numpy as np
import tensorflow as tf
from model.vgg import *
import color


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            sess,
            dim_action,
            dim_vgg_features,
            dim_lab_features,
            learning_rate,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=64,

    ):
        self.dim_action = dim_action
        self.dim_vgg_features = dim_vgg_features
        self.dim_lab_features = dim_lab_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.queue = tf.RandomShuffleQueue(self.memory_size, 1000,
                                           [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64],
                                           [[self.feature_length], [self.feature_length], 1, 1, 1])

        # 入队n个元素
        self.pushin_memory = self.queue.enqueue_many(
            [self.current_image, self.next_image, self.reward, self.action])

        # 出队n个元素
        self.current_image_many, self.next_image_many, self.reward_many, self.action_many = self.queue.dequeue_many(
            self.batch_size)

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = sess

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def get_lab_feature(self,image):
        image = self.sess.run(image)
        image_np_rgb = image + 0.5  # 0~1 rgb
        image_np_lab = color.rgb2lab(image_np_rgb)
        num_bin_L = 10
        num_bin_a = 10
        num_bin_b = 10
        L_max = 100
        L_min = 0
        a_max = 60
        a_min = -60
        b_max = 60
        b_min = -60
        image_np_lab = image_np_lab.reshape([224 * 224, 3])
        H, edges = np.histogramdd(image_np_lab, bins=(num_bin_L, num_bin_a, num_bin_b), range=((L_min, L_max), (a_min, a_max), (b_min, b_max)))
        return H.reshape(1000) / 1000.0

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.current_image =  tf.placeholder(tf.float32, [None,224,224,3], name='state_image')  # input State
        self.next_image     = tf.placeholder(tf.float32, [None,224,224,3], name='next_image')   # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        self.vgg_feature_current, _ = model_vgg(input=self.current_image, model_path=VGG_PATH)
        self.lab_feature_current = self.get_lab_feature(self.current_image)
        self.state_current = tf.concat([self.vgg_feature_current, self.lab_feature_current], axis=1)
        with tf.variable_scope('eval_net'):
            l_1_e = tf.layers.dense(self.state_current, 4096, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='f1_e')
            l_2_e = tf.layers.dense(l_1_e, 4096, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='f2_e')
            l_3_e = tf.layers.dense(l_2_e, 512, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='f3_e')
            self.q_eval = tf.layers.dense(l_3_e, self.dim_action, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='q_e')

        # ------------------ build target_net ------------------
        self.vgg_feature_next, _ = model_vgg(input=self.next_image, model_path=VGG_PATH)
        self.lab_feature_next = self.get_lab_feature(self.next_image)
        self.state_next = tf.concat([self.vgg_feature_next, self.lab_feature_next], axis=1)
        with tf.variable_scope('target_net'):
            l_1_t = tf.layers.dense(self.state_current, 4096, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='f1_t')
            l_2_t = tf.layers.dense(l_1_t, 4096, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='f2_t')
            l_3_t = tf.layers.dense(l_2_t, 512, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='f3_t')
            self.q_next = tf.layers.dense(l_3_t, self.dim_action, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='q_t')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        self.sess.run(self.pushin_memory,feed_dict = {
            self.current_image: s[np.newaxis,:],
            self.next_image   : s_[np.newaxis, :],
            self.reward: r[np.newaxis, :],
            self.action: a[np.newaxis, :],
        })

    def get_data_from_memory(self):
        current_img,next_img,r,a = self.sess.run([self.current_image_many, self.next_image_many, self.reward_many, self.action_many])
        return current_img,next_img,r,a

    def choose_action(self, image):
        # to have batch dimension when feed into tf placeholder
        image = image[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.current_image: image})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self,learn_step_counter):
        # check to replace target parameters
        if learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        current_img,next_img,r,a = self.get_data_from_memory()

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.current_image: current_img,
                self.a: a,
                self.r: r,
                self.next_image: next_img,
            })

        self.cost_his.append(cost)
        return cost

