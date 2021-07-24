#!/usr/bin/env python3

import sys
CENTERTRACK_PATH = '/CenterTrack/src/lib/'
sys.path.insert(0, CENTERTRACK_PATH)

from detector import Detector
from opts import opts

import rospy
import message_filters
from cv_bridge import CvBridge
# from tf.transformations import quaternion_matrix
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3, Point
from ford_msgs.msg import Clusters

import os
import cv2
import numpy as np
from copy import deepcopy
import time
from datetime import datetime
from scipy.spatial.transform import Rotation


# https://en.wikipedia.org/wiki/Alpha_beta_filter
class AlphaBetaFilter():
    def __init__(self, t, x, alpha=0.3, beta=0.08):
        '''
        :param t: initial timestamp (s)
        :param x: intial position measurement [x, y] np array
        For convergence and stability:
          0 < alpha < 1
          0 < beta <= 2
          0 < 4 - 2*alpha - beta
        '''
        self.alpha = alpha
        self.beta = beta

        self.x_hat = x                   # np array of estimated position [x, y] (robot coordinates)
        # TODO: better initial velocity estimate ??
        self.v_hat = np.array([0., 0.])  # np array of estimated velocity [x, y]
        self.last_time = t               # last timestamp (s) used to estimate delta T

        # For debugging.
        self.timestamp_history = []           # all past timestamps of updates (floats)
        self.measurement_history = []         # all past position measurements [x, y] np arrays
        self.estimated_position_history = []  # all past estimated positions [x, y] np arrays
        self.estimated_velocity_history = []  # all past estiamted velocities [x, y] np arrays

    def update(self, t, x):
        '''
        Updates esimated position and velocity.
        :param t: measurement timestamp (s)
        :param x: np array [x, y] position measurement
        '''
        dt = float(t - self.last_time)
        self.last_time = t

        self.x_hat += dt * self.v_hat                                           # (1)
        r_hat = x - self.x_hat  # prediction error (residual, innovation, etc)  # (3)

        self.x_hat += self.alpha * r_hat                                        # (4)
        self.v_hat += (self.beta/dt) * r_hat                                    # (5)

        # For debugging.
        self.timestamp_history.append(t)
        self.measurement_history.append(x)
        self.estimated_position_history.append(deepcopy(self.x_hat))
        self.estimated_velocity_history.append(deepcopy(self.v_hat))

    def get_position(self):
        # Gets x, y position estimate as numpy array.
        return self.x_hat

    def get_velocity(self):
        '''
        Gets estimated velocity.
        :return: geometry_msgs/Vector3 of velocity
        '''
        x, y = self.v_hat
        return Vector3(x, y, 0)

    def get_all_poses(self):
        '''
        :return: list of arrays of timestamps, x, y, and z positions
        '''
        z = np.zeros_like(self.timestamp_history)
        p = np.vstack((np.array(self.timestamp_history), np.array(self.estimated_position_history).T, z))  # estimated z coordinate is 0
        if not p.shape[0] == 4:
            p = np.zeros((4,1))
        return p


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
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.X_WR = None
        self.X_RC = np.eye(4,4)
        self.last_poses = {}        # dictionary mapping tracking ID to AlphaBetaFilter
        self.all_pose_history = {}  # dictionary mappping tracking ID to AlphaBetaFilter of all IDs in the past.
        self.depth_images = {}  # timestamp to depth image msg
        self.last_depth_time = None
        self.depth_hits = 0
        self.depth_misses = 0

        self.output_image_pub = rospy.Publisher(self.opt.output_image_topic, Image, queue_size=1)
        self.clusters_pub = rospy.Publisher('/JA01/cluster/output/clusters', Clusters, queue_size=1)
        self.robot_pose_pub = rospy.Publisher('/JA01/pose', PoseStamped, queue_size=1)
        self.robot_vel_pub = rospy.Publisher('/JA01/velocity', Vector3, queue_size=1)

        self.odom_sub = rospy.Subscriber('/ov_msckf/odomimu', Odometry, self.odom_callback)  # state estimation module
        # self.color_sub = rospy.Subscriber(self.opt.input_image_topic, Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber(self.opt.input_depth_topic, Image, self.depth_callback)
        # Subscribe to color images and depth images together.
        color_sub = message_filters.Subscriber(self.opt.input_image_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(self.opt.input_depth_topic, Image)
        ts = message_filters.TimeSynchronizer([color_sub, depth_sub], 40)
        ts.registerCallback(self.image_callback)

    def depth_callback(self, depth_msg):
        timestamp = self.get_time(depth_msg.header)
        self.depth_images[timestamp] = depth_msg
        self.last_depth_time = timestamp

    def odom_callback(self, odom):
        self.X_WR = self.pose_to_transformation(odom.pose.pose, point=True)

        robot_pose = PoseStamped()
        robot_pose.header = odom.header
        robot_pose.pose = odom.pose.pose
        
        robot_vel = Vector3()
        robot_vel.x = odom.twist.twist.linear.x
        robot_vel.y = odom.twist.twist.linear.y

        self.robot_pose_pub.publish(robot_pose)
        self.robot_vel_pub.publish(robot_vel)

    def image_callback(self, color_msg, depth_msg):
        start_time = time.time()
        self.im_count += 1
        # if not self.im_count%2 == 0:
        #     return

        # Transforming objects from camera frame to global frame.
        if self.X_WR is None:  # give warning, but assume robot is not moving
            print('State esimation not running.')
            X_WC = self.X_RC
        else:
            X_WC = np.dot(self.X_WR, self.X_RC)

        # color_im = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        color_im = self.bridge.compressed_imgmsg_to_cv2(color_msg, 'bgr8')
        ret = self.detector.run(color_im, meta = {'calib': self.K})
        # print(ret['results'])
        # print('*** IMAGES ***:', ret['images'].keys())

        timestamp = self.get_time(color_msg.header)
        # if timestamp in self.depth_images:
        #     depth_msg = self.depth_images[timestamp]
        #     self.depth_hits += 1
        # else:
        #     if self.last_depth_time is None:
        #         return
        #     depth_msg = self.depth_images[self.last_depth_time]
        #     print("NOW TIME:", timestamp)
        #     print("LAST DEPTH:", self.last_depth_time)
        #     # print(self.depth_images.keys())
        #     self.depth_misses += 1
        # print("DEPTH HITS : MISSES --", self.depth_hits, ':', self.depth_misses)
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')  # units: mm
        rr, cc = depth_im.shape

        clusters = Clusters()
        clusters.header.stamp = color_msg.header.stamp
        clusters.header.frame_id = 'world'
        all_poses = {}  # to replace self.last_poses
        # skipped_ids = []
        if ret['results']:
            for o in ret['results']:
                tc, tr, bc, br = [int(i) for i in o['bbox'].tolist()]
                if (bc - tc < 50 or br - tr < 50) and br < 240:  # skip if a camera on the ceiling is a detected as a person
                    # skipped_ids.append(o['tracking_id'])
                    continue

                # Depths within center region to top of bbox.
                width, height = bc - tc, br - tr  # width, height of bbox in pixels
                v_center, u_center = int((tr + br)//2), int((tc + bc)//2)  # bbox center coordinates (row, col)
                v_width, u_width = int(height//10), int(width//10)  # height and width of center box to extract depth
                center_depths = depth_im[max(0, v_center-v_width):min(rr, v_center+v_width+1), max(0, u_center-u_width):min(cc, u_center+1)].flatten()

                # Use median of center strip for depth.
                # Center strip should not be fully occluded because it wouldn't be detected.
                # In most cases, should be majority pedestrian of interest with a small portion of
                # background pixels and occlusion pixels.
                center_depths = sorted(center_depths)[len(center_depths)//4:]  # getting rid of bottom half of depths to hope to exclude occlusions

                # Positions (mm) in camera coordinates.
                Z = center_depths[len(center_depths)//2]  # approx. median because center_depths already sorted
                v, u = int(br), int((tc + bc)//2.) 
                X = self.u_to_x(u, Z)
                Y = self.v_to_y(v, Z)

                t_id = o['tracking_id']
                clusters.labels.append(t_id)
                clusters.radii.append(Float32(0.3))  # hard-coded pedestrian radius (m)

                Z /= np.cos(0.3)    # camera is pitched up relative to the robot frame
                X_CO = np.array([[Z/1000.],[-X/1000.],[-Y/1000.],[1]])  # position of object in robot coordinate system but still in camera frame
                X_WO = np.dot(X_WC, X_CO)

                pose_array = np.concatenate(([timestamp], X_WO.reshape(4,)[:-1]))
                if t_id in self.last_poses:
                    pose_history = self.last_poses[t_id]
                    pose_history.update(timestamp, pose_array[1:-1])
                    all_poses[t_id] = pose_history
                else:
                    all_poses[t_id] = AlphaBetaFilter(timestamp, pose_array[1:-1])

                est_x, est_y = all_poses[t_id].get_position()
                clusters.poses.append(Point(est_x, est_y, 0))
                clusters.velocities.append(all_poses[t_id].get_velocity())   

                if self.output_image_pub.get_num_connections() > 0:
                    generic_im = self.bridge.cv2_to_imgmsg(ret['images']['generic'], encoding='bgr8')
                    generic_im.header.stamp = color_msg.header.stamp
                    self.output_image_pub.publish(generic_im)

            self.last_poses = all_poses
            self.all_pose_history.update(self.last_poses)
        self.clusters_pub.publish(clusters)  # TODO: indent this if you don't want to publish empty clusters

        print("TOTAL TIME (s):", time.time() - start_time)

        # print('[track_node]  SKIPPED IDs: %s'%(str(skipped_ids)))

        # print('======================================')

    def u_to_x(self, u, Z):
        # Converts u pixel coordinate to x position in meters.
        return (u - self.cx)*Z/float(self.fx)

    def v_to_y(self, v, Z):
        # Converts v pixel coordinate to y position in meters.
        return (v - self.cy)*Z/float(self.fy)

    def get_time(self, header):
        '''
        Calculates the time stamp of the image in seconds.
        :param header: std_msgs/Header instance
        '''
        return header.stamp.secs + header.stamp.nsecs*10**(-9)

    def pose_to_transformation(self, pose, point=True):
        '''
        Converts a geometry_msgs/Pose message to a transformation matrix.
        :param pose: geometry_msgs/Pose
        :param point: If true, return a rotation + translation transformation matrix. If false, only use rotation.
        :return: homogeneous transformation matrix as 4x4 numpy array
        '''
        q = pose.orientation
        trans = np.eye(4, 4)
        trans[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()  # 4x4 rotation transformation matrix
        if not point:
            return trans
        p = pose.position
        trans[:3, 3] = np.array([p.x, p.y, p.z])  # 4x4 transformation matrix with rotation + translation
        return trans

    def point_to_homogeneous(self, point):
        '''
        Converts a point to a homogeneous vector representation.
        :param point: geometry_msgs/Point or geometry_msgs/Vector3 or any message with x, y, and z components
        :return: 4x1 homogeneous vector representation of the point
        '''
        return np.array([[point.x],
                         [point.y],
                         [point.z],
                         [      1]])

    def homogeneous_to_point(self, matrix, point=True):
        '''
        Converts a homogeneous matrix to a point or vector in 3D space.
        :param matrix: 4x3 homogeneous matrix
        :param point: if True, use geometry_msgs/Point or else use geometry_msgs/Vector message
        :return: geometry_msgs/Point or geometry_msgs/Vector with x,y,z components filled
        '''
        xyz = matrix[:3]/float(matrix[3])
        if point:
            return Point(*xyz)
        else:
            return Vector3(*xyz)


if __name__ == '__main__':
    rospy.init_node("centertrack_node")
    node = CenterTrackNode()
    rospy.spin()