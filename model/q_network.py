import tensorflow as tf
import numpy as np
from model.vgg import *

class Q_Net():

    def init(self,input_tensor, input_feature_length=512, num_action=11):

        fc1_weights = tf.get_variable("fc1_weights", [input_feature_length, 4096],
                                      initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc1_biases = tf.get_variable("fc1_biases", [4096],
                                     initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc2_weights = tf.get_variable("fc2_weights", [4096, 4096],
                                      initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc2_biases = tf.get_variable("fc2_biases", [4096],
                                     initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc3_weights = tf.get_variable("fc3_weights", [4096, 4096],
                                      initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc3_biases = tf.get_variable("fc3_biases", [4096],
                                     initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc4_weights = tf.get_variable("fc4_weights", [4096, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 4096)))
        fc4_biases = tf.get_variable("fc4_biases", [512],
                                     initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 512)))
        fc5_weights = tf.get_variable("fc5_weights", [512, num_action],
                                      initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / 512)))
        fc5_biases = tf.get_variable("fc5_biases", [num_action],
                                     initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1.0 / num_action)))
        weights = {}
        weights['fc1_weights'] = fc1_weights
        weights['fc1_biases'] = fc1_biases
        weights['fc2_weights'] = fc2_weights
        weights['fc2_biases'] = fc2_biases
        weights['fc3_weights'] = fc3_weights
        weights['fc3_biases'] = fc3_biases
        weights['fc4_weights'] = fc4_weights
        weights['fc4_biases'] = fc4_biases
        weights['fc5_weights'] = fc5_weights
        weights['fc5_biases'] = fc5_biases
        tensor = tf.nn.dropout(input_tensor, 1.0)
        tensor = tf.nn.relu(tf.matmul(tensor, fc1_weights) + fc1_biases)
        tensor = tf.nn.relu(tf.matmul(tensor, fc2_weights) + fc2_biases)
        tensor = tf.nn.relu(tf.matmul(tensor, fc3_weights) + fc3_biases)
        tensor = tf.nn.relu(tf.matmul(tensor, fc4_weights) + fc4_biases)
        tensor = tf.matmul(tensor, fc5_weights) + fc5_biases
        return tensor, weights

    # def save_model(self, sess,step=None):
    #     print(" [*] Saving checkpoints...")
    #     model_name = type(self).__name__
    #     self.saver.save(sess, "./checkpoints/" + self.prefix + ".ckpt", global_step=step)
    #
    # def load_model(self, model_path):
    #     self.saver.restore(sess, model_path)
    #     print(" [*] Load success: %s" % model_path)
    #     return True

