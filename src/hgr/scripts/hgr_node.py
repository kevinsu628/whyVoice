#!/usr/bin/env python

import numpy as np
import threading
from keras.models import load_model
from kws.msg import kwsData
import cv2
import time
import rospy


class HGRNode:
    def __init__(self, model_path):
        rospy.init_node('HGRNode', anonymous=True)
        self.labels = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
                       "Pushing Hand Away", "Pulling Hand In", "Sliding Two Fingers Left", "Sliding Two Fingers Right",
                       "Sliding Two Fingers Down", "Sliding Two Fingers Up", "Pushing Two Fingers Away",
                       "Pulling Two Fingers In",
                       "Rolling Hand Forward", "Rolling Hand Backward", "Turning Hand Clockwise",
                       "Turning Hand Counterclockwise",
                       "Zooming In With Full Hand", "Zooming Out With Full Hand", "Zooming In With Two Fingers",
                       "Zooming Out With Two Fingers",
                       "Thumb Up", "Thumb Down", "Shaking Hand", "Stop Sign",
                       "Drumming Fingers", "No Gesture", "Doing Other Things"]
        self.label2label = {
            "Zooming In With Full Hand": "Turn on the light",
            "Zooming In With Two Fingers": "Turn on the light",
            "Zooming Out With Full Hand": "Turn off the light",
            "Zooming Out With Two Fingers": "Turn off the light",
            "Swiping Left": "Dim the light",
            "Sliding Two Fingers Left": "Dim the light",
            "Swiping Right": "Brighten the light",
            "Sliding Two Fingers Right": "Brighten the light"
        }
        self.idx_to_label = {i: self.labels[i] for i in range(27)}
        self.hgr_lock = threading.RLock()
        self.model = load_model(model_path)
        self.FRAME_SIZE = (96, 64)
        self.N_FRAME = 16
        self.last_kws = 0
        self.conf_thresh = 0.8
        self.kw_idx = 2  # keyword index
        self.time_stop = 8  # seconds for the camera to be turned off if no hand gesture detected
        self.audio_sub = rospy.Subscriber('/kws',
                                          kwsData,
                                          callback=self.audio_callback,
                                          queue_size=1,
                                          buff_size=1)
        rospy.loginfo("HGR pretrained model loaded")

    def audio_callback(self, kwsData):
        pred_idx = kwsData.cls
        pred_conf = kwsData.conf
        if pred_idx == self.kw_idx and pred_conf > self.conf_thresh and self.last_kws != self.kw_idx:
            rospy.loginfo("Calling camera")
            self.camera_on_and_predict()
        self.last_kws = pred_idx

    def camera_on_and_predict(self):
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        winname = "frame"
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname, 600, 280)  # Move it to (40,30)
        buffer = []
        final_label = ""
        detect_predefined = False
        i = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                i += 1
                # resize to match training sample
                image = cv2.resize(frame, self.FRAME_SIZE)
                # normalize
                image = image / 255
                buffer.append(image)
                if i % self.N_FRAME == 0:  # one complete instance
                    buffer = np.expand_dims(buffer, 0)
                    predicted_value = np.argmax(self.model.predict(buffer))
                    final_label = self.idx_to_label[predicted_value]
                    detect_predefined = (final_label in self.label2label.keys())
                    if detect_predefined:
                        final_label = self.label2label[final_label]
                    cv2.imshow(winname, frame)
                    buffer = []
                text = final_label if detect_predefined else ""
                cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
                cv2.imshow(winname, frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or detect_predefined or i // self.N_FRAME == self.time_stop:
                break
        time.sleep(1)
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        hgr_model_path = "/home/lang/Ascent/project/whyvoice/src/hgr/model/resnetmodel.hdf5"
        HGRNode(hgr_model_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
