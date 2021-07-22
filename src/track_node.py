#!/usr/bin/env python3

import sys
CENTERTRACK_PATH = '/CenterTrack/src/lib/'
sys.path.insert(0, CENTERTRACK_PATH)

from detector import Detector
from opts import opts

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ford_msgs.msg import CenterTrackObjects, CenterTrackObject

import os
import cv2
import numpy as np

class CenterTrackNode():
    def __init__(self):
        self.model_path = '/CenterTrackModels/models/crowdhuman.pth'
        self.task = 'tracking,multi_pose'
        # --show_track_color   have track ID on bbox label instead of confidence score
        # --no_pause           if --debug > 0, show images continually instead of having to click a key to advance
        # --ros                ROS implementation
        other = '--dataset crowdhuman --input_h 480 --input_w 640 --track_thresh 0.3 --show_track_color --no_pause --ros'  # add '--debug 1' to view output
        # other += ' --input_image_topic /camera/color/image_raw'
        other += ' --input_image_topic /camera_d455/color/image_raw/compressed'
        self.opt = opts().init('{} --load_model {} {}'.format(self.task, self.model_path, other).split(' '))
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str

        self.bridge = CvBridge()
        self.detector = Detector(self.opt)
        self.im_count = -1

        # Home color camera
        # self.K = np.array([[549.1329067091865, 0.0, 318.65780926484354, 0.0],
                           # [0.0, 551.9781059645037, 259.6421848387817, 0.0],
                           # [0.0, 0.0, 1.0, 0.0]])
        # Jackal d435i color camera
        # self.K = np.array([[599.2294541560864, 0.0, 335.2098573031863, 0.0],
        #                    [0.0, 598.0449686657284, 256.34181912731594, 0.0],
        #                    [0.0, 0.0, 1.0, 0.0]])
        # Jackal d455 color camera
        self.K = np.array([[383.09075523860304, 0.0, 312.84915584333794, 0.0],
                           [0.0, 381.5605199396851, 244.79360424448808, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])

        self.output_image_pub = rospy.Publisher(self.opt.output_image_topic, Image, queue_size=1)
        self.track_output_pub = rospy.Publisher(self.opt.track_output_topic, CenterTrackObjects, queue_size=1)

        self.input_image_sub = rospy.Subscriber(self.opt.input_image_topic, CompressedImage, self.image_callback)
        # self.input_image_sub = rospy.Subscriber(self.opt.input_image_topic, Image, self.image_callback)

    def image_callback(self, image):
        self.im_count += 1
        if not self.im_count%5 == 0:
            return

        img = self.bridge.compressed_imgmsg_to_cv2(image, 'bgr8')
        # img = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        ret = self.detector.run(img, meta = {'calib': self.K})
        print(ret['results'])
        print('*** IMAGES ***:', ret['images'].keys())

        timestamp = image.header.stamp
        objects = CenterTrackObjects()
        objects.header.stamp = timestamp
        skipped_ids = []
        if ret['results']:
            for o in ret['results']:
                tc, tr, bc, br = [int(i) for i in o['bbox'].tolist()]
                if (bc - tc < 50 or br - tr < 50) and br < 240:  # skip if a camera on the ceiling is a detected as a person
                    skipped_ids.append(o['tracking_id'])
                    continue

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

            self.output_image_pub.publish(generic_im)
        self.track_output_pub.publish(objects)
        print('[track_node]  SKIPPED IDs: %s'%(str(skipped_ids)))

        print('======================================')


if __name__ == '__main__':
    rospy.init_node("centertrack_node")
    node = CenterTrackNode()
    rospy.spin()