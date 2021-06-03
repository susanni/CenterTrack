#!/usr/bin/env python3

import sys
CENTERTRACK_PATH = '/CenterTrack/src/lib/'
sys.path.insert(0, CENTERTRACK_PATH)

from detector import Detector
from opts import opts

import rospy
# from ford_msgs.msg import CenterTrackObjects, CenterTrackObject

import os
import cv2
import numpy as np

# def CenterTrackNode():
# 	def __init__(self):
# 		self.model_path = '/CenterTrackModels/models/crowdhuman.pth'
# 		self.task = 'tracking,multi_pose'
# 		other = '--dataset crowdhuman --input_h 480 --input_w 640 --track_thresh 0.3 --no_pause'  # add '--debug 1' to view output
# 		self.opt = opts().init('{} --load_model {} {}'.format(self.task, self.model_path, other).split(' '))
# 		os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str

# 		self.detector = Detector(self.opt)

# 		self.images_dir = '/CenterTrackModels/image_sets/office_middle/'


# if __name__ == '__main__':
# 	rospy.init_node("center_track_node")
# 	node = CenterTrackNode()
# 	rospy.spin()



MODEL_PATH = '/CenterTrackModels/models/crowdhuman.pth'
# TASK can be 'tracking', 'tracking,multi_pose' for pose tracking, or 'tracking,ddd' for monocular 3d tracking.
TASK = 'tracking,multi_pose'
other = '--dataset crowdhuman --input_h 480 --input_w 640 --track_thresh 0.3 --no_pause'  # add '--debug 1' to view output
opt = opts().init('{} --load_model {} {}'.format(TASK, MODEL_PATH, other).split(' '))
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
opt.debug = max(opt.debug, 1)
detector = Detector(opt)

IMAGES_DIR = '/CenterTrackModels/image_sets/office_middle/'
images = sorted(os.listdir(IMAGES_DIR))
for im_name in images:
	img = cv2.imread(IMAGES_DIR + im_name)
	# Last column of K is camera extrinsics?
	K = np.array([[549.1329067091865, 0.0, 318.65780926484354, 0.0],
                  [0.0, 551.9781059645037, 259.6421848387817, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
	ret = detector.run(img, meta = {'calib': K})
	print(ret['results'])
	print('======================================')