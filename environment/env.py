import numpy as np
from utils.env_op import compute_color_l2,compute_black_l2
import os
import random
from utils.load_image import *
from utils.retouch_op import *
from utils.train_op import *
import time
import glob
from prepare_data_pair.actions import *


class Environment(object):

    def __init__(self,raw_path,target_path,whether_baseline):
        self.raw_path = raw_path  # 脏数据
        self.target_path = target_path  # 修饰完的理想状态
        self.source_image = None
        self.target_image = None
        self.current_image = None
        self.l2_distance_old = None
        self.raw_name_list = glob.glob(self.raw_path+"*.jpg")
        self.target_name_list = glob.glob(self.target_path+"*.jpg")
        if whether_baseline:
            self.action_list = None
            self.dim_a = None
        else:
            self.sub_action_list = sub_action_list
            self.master_action_list = master_action_list
            self.dim_s = None
            self.dim_a_master = len(master_action_list)
            self.dim_a_sub = len(sub_action_list)
        self.action_trajectory = []
        self.done = False



    def reset(self):
        raw_img_path = random.choice(self.raw_path)
        raw_img_name = os.path.basename(raw_img_path)
        raw_img_index = raw_img_name.split('_')[0]
        self.raw_image = load_image(raw_img_path)
        self.target_image = load_image( self.target_path+"%s.jpg"%raw_img_index) # 干净的目标图像
        self.current_image = self.raw_image
        self.l2_distance_old = compute_color_l2(current_image=self.current_image,target_image=self.target_image)
        self.action_trajectory = []
        self.done = False
        return self.current_image


    def take_action(self,action_index):
        self.action_trajectory.append(action_index)
        result = take_action(self.current_image,action_index)
        new_l2_distance = compute_color_l2(current_image=result, target_image=self.target_image)
        reward = self.l2_distance_old - new_l2_distance
        self.current_image = result
        if new_l2_distance < TERMINAL_THRESHOLD:
            self.done = True
        return result,reward,self.done



    def take_sub_action(self,action_index):
        self.action_trajectory.append(self.sub_action_list[action_index])
        result = self.sub_action_list[action_index](self.current_image)
        new_l2_distance = compute_color_l2(current_image=result, target_image=self.target_image)
        reward = self.l2_distance_old - new_l2_distance
        self.l2_distance_old = new_l2_distance
        self.current_image = result
        #print("sub_distance: ",new_l2_distance)
        if new_l2_distance < TERMINAL_THRESHOLD:
            self.done = True
        # result = np.clip(result*255.0,a_min=0.0,a_max=255.0).astype(np.uint8)
        return result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done

    def take_master_action(self, action_index):
        self.action_trajectory.append(self.master_action_list[action_index])
        result = self.master_action_list[action_index](self.current_image)
        new_l2_distance = compute_color_l2(current_image=result, target_image=self.target_image)
        reward = self.l2_distance_old - new_l2_distance
        self.l2_distance_old = new_l2_distance
        self.current_image = result
        #print("master_distance: ",new_l2_distance)
        if new_l2_distance < TERMINAL_THRESHOLD:
            self.done = True
        # result = np.clip(result*255.0,a_min=0.0,a_max=255.0).astype(np.uint8)
        return result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done

    def show(self):
        show_image(self.current_image)

    def show_target(self):
        show_image(self.target_image)

    def get_action_trajectory(self):
        return  self.action_trajectory





    def get_color_feature(self,image):
        channel_one,channel_two,channel_three = distribution_color(image)
        result = np.concatenate([channel_one,channel_two,channel_three],axis=0).reshape([1,-1])[0]
        # print("color_feature:",result)
        return result

    def get_gray_feature(self,image):
        result = distribution_gray(image).reshape([1,-1])[0]
        # print("gray_feature:",result)
        return result

    def save_env_image(self,success,epi):
        time_current = time.strftime("[%Y-%m-%d-%H-%M-%S]", time.localtime(time.time()))
        if success:
            save_image(self.current_image, filepath=result_save_path +str(epi)+'_'+'Success_'+ str(time_current) + '_No:'+ self.img_name )
        else:
            save_image(self.current_image, filepath=result_save_path +str(epi)+'_'+'False_'  + str(time_current) + '_No:'+ self.img_name )




if __name__ == "__main__":
       env = Environment(target_path=root + 'data/source_data/',source_path=root + 'data/train_data/')
       env.reset()