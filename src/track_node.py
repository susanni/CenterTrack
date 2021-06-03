#!/usr/bin/env python3

import sys
CENTERTRACK_PATH = '/CenterTrack/src/lib/'
sys.path.insert(0, CENTERTRACK_PATH)

from detector import Detector
from opts import opts

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ford_msgs.msg import CenterTrackObjects, CenterTrackObject

import os
import cv2
import numpy as np

class CenterTrackNode():
	def __init__(self):
		self.model_path = '/CenterTrackModels/models/crowdhuman.pth'
		self.task = 'tracking,multi_pose'
		other = '--dataset crowdhuman --input_h 480 --input_w 640 --track_thresh 0.3 --no_pause --ros'  # add '--debug 1' to view output
		self.opt = opts().init('{} --load_model {} {}'.format(self.task, self.model_path, other).split(' '))
		os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str

		self.bridge = CvBridge()
		self.detector = Detector(self.opt)

		self.images_dir = '/CenterTrackModels/image_sets/office_middle/'
		self.images = sorted(os.listdir(self.images_dir))
		self.K = np.array([[549.1329067091865, 0.0, 318.65780926484354, 0.0],
                           [0.0, 551.9781059645037, 259.6421848387817, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])

		self.input_image_pub = rospy.Publisher(self.opt.input_image_topic, Image, queue_size=1)
		self.output_image_pub = rospy.Publisher(self.opt.output_image_topic, Image, queue_size=1)
		self.track_output_pub = rospy.Publisher(self.opt.track_output_topic, CenterTrackObjects, queue_size=1)

	def process_images(self):
		for im_name in self.images:
			if rospy.is_shutdown(): return
			img = cv2.imread(self.images_dir + im_name)
			ret = self.detector.run(img, meta = {'calib': self.K})
			print(ret['results'])
			print('*** IMAGES ***:', ret['images'].keys())

			if ret['results']:
				timestamp = rospy.get_rostime()
				objects = CenterTrackObjects()
				objects.header.stamp = timestamp
				for o in ret['results']:
					obj = CenterTrackObject()
					obj.score = o['score']
					obj.id = o['tracking_id']
					obj.class_id = o['class']
					obj.ct = o['ct'].tolist()
					obj.tracking = o['tracking'].tolist()
					obj.bbox = o['bbox'].tolist()
					obj.age = o['age']
					obj.active = o['active']
					objects.objects.append(obj)

				previous_im = self.bridge.cv2_to_imgmsg(ret['images']['previous'], encoding='bgr8')
				generic_im = self.bridge.cv2_to_imgmsg(ret['images']['generic'], encoding='bgr8')
				previous_im.header.stamp = timestamp
				generic_im.header.stamp = timestamp

				self.input_image_pub.publish(previous_im)
				self.output_image_pub.publish(generic_im)
				self.track_output_pub.publish(objects)

			print('======================================')


if __name__ == '__main__':
	rospy.init_node("centertrack_node")
	node = CenterTrackNode()
	node.process_images()
	# rospy.spin()



# MODEL_PATH = '/CenterTrackModels/models/crowdhuman.pth'
# # TASK can be 'tracking', 'tracking,multi_pose' for pose tracking, or 'tracking,ddd' for monocular 3d tracking.
# TASK = 'tracking,multi_pose'
# other = '--dataset crowdhuman --input_h 480 --input_w 640 --track_thresh 0.3 --no_pause'  # add '--debug 1' to view output
# opt = opts().init('{} --load_model {} {}'.format(TASK, MODEL_PATH, other).split(' '))
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
# opt.debug = max(opt.debug, 1)
# detector = Detector(opt)

# IMAGES_DIR = '/CenterTrackModels/image_sets/office_middle/'
# images = sorted(os.listdir(IMAGES_DIR))
# for im_name in images:
# 	img = cv2.imread(IMAGES_DIR + im_name)
# 	# Last column of K is camera extrinsics?
# 	K = np.array([[549.1329067091865, 0.0, 318.65780926484354, 0.0],
#                   [0.0, 551.9781059645037, 259.6421848387817, 0.0],
#                   [0.0, 0.0, 1.0, 0.0]])
# 	ret = detector.run(img, meta = {'calib': K})
# 	print(ret['results'])
# 	print(ret['images'])  # generic -- output, previous -- input
# 	print('======================================')