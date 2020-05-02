import tensorflow as tf
from model.model_dqn import DeepQNetwork
from environment.env import *
from config.config import *



class DQN_Worker(object):

    def __init__(self,
                 name,
                 sess,
                 ):


        self.name = name

        self.env = Environment(raw_path=RAW_DATA_PATH,target_path=TARGET_DATA_PATH,whether_baseline=False)

        self.net = DeepQNetwork(
            sess = sess,
            dim_action=12,
            dim_vgg_features=" ",
            dim_lab_features=" ",
            learning_rate=0.00001,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=64)

        self.step_count = 0
        self.episode_count = 0



    def work(self):

        while self.episode_count <= EP_MAX:
            EP_COUNT = EP_COUNT + 1
            ep_r  = 0
            steps = 0
            # result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done
            current_img = env.reset()
            while True:
                # fresh env

                # RL choose action based on observation
                action = self.net.choose_action(current_img)

                # RL take action and get next observation and reward
                next_img, reward, done = self.env.step(action)

                ep_r = ep_r + reward

                steps = steps + 1

                self.step_count = self.step_count + 1

                self.net.store_transition( current_img, action, reward, next_img)

                if (self.step_count > 500) and (self.step_count % 5 == 0):
                    loss = self.net.learn()
                    print("epi:%s  loss:%s"%(EP_COUNT,loss))

                # swap observation
                current_img = next_img

                # break while loop when end of this episode
                if done or steps>=20:
                    break


if __name__ == "__main__":

    with tf.device('/gpu:0'):

        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        worker = DQN_Worker(name="dqn_worker",sess=SESS)

        worker.work()










