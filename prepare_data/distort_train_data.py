import random
from random import shuffle
import skimage.color as color
import os, glob, random, sys, math
import numpy as np
from action.action_for_distort import *
from util_my.img_utils import *
from config.config import *

distort_action_list = [
                    brightness,
					color_saturation,
					contrast,
					hl_brightness,
					shadow_brightness,
					hl_saturation,
					shadow_saturation,
					red_adjust_red,
					green_adjust_green,
					blue_adjust_blue,
					cyan_adjust_cyan,
					magenta_adjust_magenta,
					yellow_adjust_yellow
]


def take_action(image_np, idx, degree):
	return_np = distort_action_list[idx](image_np+0.5, degree)
	return return_np-0.5


def distort_a_image(raw_image, lower_b=0.001, higher_b=2.0, distort_single=False, use_threshold=True):
	"""
	输入一张raw_image 返回一张image,注意此处的raw_image是已经resize过的
	当distort_single为True时，用来distort的动作只有一个
	当distort_single为False时，用来distort的动作有N个，N最大为len(action_list)
	当use_threshold为True时，使用lower_b, higher_b对distort的结果的mse进行限制，如果mse太大则不返回

	返回值 image, True, mse, actions
		True时返回的结果可以正常使用
		mse是distort前后的图片误差
		action是一个列表，存放distort的轨迹，每个轨迹的格式为（action_index,动作幅度）
	"""
	actions = []
	# MULTIPLE
	# SINGLE
	if distort_single:
	#  just take the action once : len(actions) = 1
		random_idx = random.randint(0, len(distort_action_list)-1)
		sign = (random.random()<0.5)*2-1  # [1,-3]
		degree = random.uniform(0.1, 0.3) #        [-0.3,+0.3]
		actions.append((random_idx, 1+degree*sign))   # [0.7,1.3]
	else:
	# take the action fo many times : len(actions) <= len(ations_lists)
		for i in range(int(len(distort_action_list))):
			# action_lists当中的动作有50%的概率被选中进入actions，同时action的degree由 1+degree*sign 确定
			# action_degree = 1 + (0.1~0.2之间的均匀分布)*(1或者-1)
			if random.random() > 0.7:#apply random distortion from 0.6~1.0
				sign = (random.random()<0.5)*2-1
				degree = random.uniform(0.1, 0.2)
				actions.append((i, 1+degree*sign))
	# 打乱actions
	shuffle(actions)
	image = raw_image.copy()
	#action_str = ""

	# 根据actions ， 对image采取actions，生成新的image
	for action_pair in actions:  # action pair = action---degree
		idx, degree = action_pair
		image = take_action(image, idx, degree)
		#action_str += "%d_%.2f_" % (idx, float(degree))

	raw_image_lab = color.rgb2lab(raw_image+0.5) # 原图
	image_lab     = color.rgb2lab(image+0.5)     # 人为改变后的图片
	#mse = (( image_lab - raw_image_lab )**2).mean()/100
	mse = np.sqrt(np.sum(( raw_image_lab - image_lab)**2, axis=2)).mean()/10.0

	if use_threshold:
		if mse >= lower_b and mse < higher_b:
			return image, True, mse, actions
		else:
			return None, False, mse, actions
	else:
		return image, True, mse, actions


def distort_train_images(train_data_path,save_data_path):
	raw_img_list = glob.glob(train_data_path)
	if not os.path.exists(save_data_path):
		os.mkdir(save_data_path)
	for item in raw_img_list:
		item_name = os.path.basename(item)
		print(item_name)

		for i in range(10):
			img = imresize(imread(item), (224,224))/255.0-0.5
			distorted_image, done, mse, actions = distort_a_image(img, distort_single=False,use_threshold=True)
			if done:
				distorted = Image.fromarray(np.uint8(np.clip((distorted_image + 0.5) * 255, 0, 255)))
				raw       = Image.fromarray(np.uint8(np.clip((img + 0.5) * 255, 0, 255)))
				item_name = os.path.basename(item).split(".")[0]
				distorted.save(os.path.join(save_data_path, '%s__%f.jpg' % (os.path.basename(item_name), mse)))
				raw.save(os.path.join(save_data_path, '%s__%s.jpg' % (os.path.basename(item_name), 'raw')))
			else:
				print("fail : %s"%mse)
	return


if __name__ == '__main__':

	# distort_train_images(train_data_path = DATA_PATH + "/train/raw/*.jpg",
	# 					 save_data_path  = DATA_PATH + "train/distort_raw/")

	a = glob.glob( DATA_PATH + "/train/raw/000000.jpg")
	print(a)




