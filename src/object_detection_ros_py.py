#!/usr/bin/python2
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from detect_image import DetectImage


class DetectVideo(object):
    # parameters need to modify
    # node
    subscribed_topic = '/my_video'
    # video
    show_video_flag = True
    save_video_flag = False
    video_rate = 30.0
    video_name = 'out.avi'
    # detect
    # Create DetectImage class
    OBJECT_DETECTION_PATH = '/home/zj/program/models/object_detection'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(OBJECT_DETECTION_PATH, 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb')
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_PATH, 'data/mscoco_label_map.pbtxt')
    NUM_CLASSES = 90


    # parameters do not need to modify
    # node
    # image_sub_ = rospy.Subscriber()
    # video
    VIDEO_WINDOW_NAME = ''
    image_hight = -1
    image_width = -1
    video = cv2.VideoWriter()
    video_file_path = os.path.join(os.path.abspath('./video'), video_name)
    # frame
    frame_num = 1
    src_3 = np.array([])
    dst_3 = np.array([])
    cvi = CvBridge()
    # detect
    di = 'DetectImage'

    def __init__(self):
        # node
        self.image_sub_ = rospy.Subscriber(self.subscribed_topic, Image, self.image_callback, queue_size=1)
        # video
        if self.show_video_flag:
            self.VIDEO_WINDOW_NAME = 'video'
            cv2.namedWindow(self.VIDEO_WINDOW_NAME)
        # detect
        self.di = DetectImage(self.PATH_TO_CKPT, self.PATH_TO_LABELS, self.NUM_CLASSES)

    def __del__(self):
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        try:
            self.src_3 = self.cvi.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return
        if self.frame_num == 1:
            self.image_hight, self.image_width, channels = self.src_3.shape
            self.video = cv2.VideoWriter(self.video_file_path, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), self.video_rate,
                                         (self.image_width, self.image_hight))
        self.frame_num += 1

        # detect
        # self.src_3 = cv2.resize(self.src_3,(160, 120))
        self.dst_3 = self.di.run_detect(self.src_3)[0]

        # save and show video
        if self.save_video_flag:
            self.video.write(self.dst_3)
        if self.show_video_flag:
            cv2.imshow(self.VIDEO_WINDOW_NAME, self.dst_3)
            cv2.waitKey(1)


if __name__ == '__main__':
    print('opencv: ' + cv2.__version__)
    rospy.init_node('object_detection_ros_py', anonymous=True)
    mrd = DetectVideo()
    rospy.spin()
