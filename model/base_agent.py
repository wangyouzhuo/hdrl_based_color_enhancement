import tensorflow as tf
import pandas as pd
import numpy as np
from model.q_network import *

class BaseAgent():

    def __init__(self,sess):

        # 用来 入队 的placeholder
        self.s_t_single = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t_single")
        self.s_t_plus_1_single = tf.placeholder('float32', [self.batch_size, self.feature_length],                                         name="s_t_plus_1_single")
        self.action_single = tf.placeholder('int64', [self.batch_size, 1], name='action_single')
        self.terminal_single = tf.placeholder('int64', [self.batch_size, 1], name='terminal_single')
        self.reward_single = tf.placeholder('float32', [self.batch_size, 1], name='reward_single')

        # 用来出队 并喂给模型训练的placeholder
        self.s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t")
        self.target_q_t = tf.placeholder('float32', [self.batch_size], name='target_q_t')
        self.action = tf.placeholder('int64', [self.batch_size], name='action')

        # 初始化replay buffer 并初始化入队 出队 的操作
        self.queue = tf.RandomShuffleQueue(self.memory_size, 1000,[tf.float32, tf.float32, tf.float32, tf.int64, tf.int64],[[self.feature_length], [self.feature_length], 1, 1, 1])
        self.enqueue_op = self.queue.enqueue_many([self.s_t_single, self.s_t_plus_1_single, self.reward_single, self.action_single, self.terminal_single])   # 入队的操作
        self.s_t_many, self.s_t_plus_1_many, self.reward_many, self.action_many, self.terminal_many = self.queue.dequeue_many(self.batch_size)  # 出队操作

        # 用于前向传播 计算q-value的placeholder
        self.target_s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="target_s_t")

        self.sess = sess

        self.target_q, self.target_q_w = Q_Net(self.target_s_t, input_length=self.feature_length,num_action=self.action_size)  # resulting q values....

    def learning_with_minibatch(self):
        #　出队若干元素进行训练
        state, reward, action, next_state, terminal = self.sess.run([self.s_t_many, self.reward_many, self.action_many, self.s_t_plus_1_many, self.terminal_many])
        q_t_plus_1 = self.target_q.eval({self.target_s_t: next_state}, session=self.sess)
        terminal_np = np.array(terminal) + 0.
        max_q_t_plus_1 = np.reshape(np.max(q_t_plus_1, axis=1), terminal_np.shape)  # next state's maximum q value
        target_q_t = (1 - terminal_np) * self.discount * max_q_t_plus_1 + reward
        action = np.reshape(action, target_q_t.shape)
        action = np.reshape(action, (self.batch_size))
        target_q_t = np.reshape(target_q_t, (self.batch_size))
        _, q_t, loss, delta, one_hot = self.sess.run([self.optim, self.q, self.loss, self.delta, self.action_one_hot],
                                                    feed_dict={	self.target_q_t	: target_q_t,
                                                                   self.action		: action,
                                                                   self.s_t		: state })
