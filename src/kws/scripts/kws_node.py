#!/usr/bin/env python

import rospy
from kws.msg import kwsData
import librosa
import torch
import torch.nn.functional as F
from collections import ChainMap
import pyaudio

import base64
import threading
import zlib

import model as mod
import numpy as np


class KWSNode(object):
    def __init__(self, kws_model_path):
        rospy.init_node('KWSNode', anonymous=True)
        self.audio_lock = threading.RLock()

        self.chunk_size = 1000
        self._audio = pyaudio.PyAudio()
        self._audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                         frames_per_buffer=self.chunk_size, stream_callback=self._on_audio)
        self.last_data = np.zeros(1000)
        self._audio_buf = []

        self.audio_pub = rospy.Publisher('/kws', kwsData, queue_size=1)
        self.out = None
        self.audio_buf = []
        self.stride_size = 500
        self.labels = ["_silence_", "_unknown_", "What's up!", "background", "Someone talking"]
        self.model = self.model_init(kws_model_path)
        rospy.loginfo("KWS pretrained model loaded")
        rospy.Timer(rospy.Duration(0.05), self.get_instances)
        rospy.loginfo("KWSNode started")

    def model_init(self, model_path):
        config = dict(dropout_prob=0.5, height=101, width=40, n_labels=5, n_feature_maps1=64,
                      n_feature_maps2=64, conv1_size=(20, 8), conv2_size=(10, 4), conv1_pool=(2, 2),
                      conv1_stride=(1, 1),
                      conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True),
        model = mod.SpeechModel(ChainMap(*config))
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        return model

    def _on_audio(self, in_data, frame_count, time_info, status):
        data_ok = (in_data, pyaudio.paContinue)
        self.last_data = in_data
        self._audio_buf.append(in_data)
        if len(self._audio_buf) != 16:
            return data_ok
        audio_data = base64.b64encode(zlib.compress(b"".join(self._audio_buf)))
        self._audio_buf = []
        wav_data = audio_data.decode()
        self.predict_and_publish(wav_data)
        return data_ok

    def predict_and_publish(self, wav_data):
        wav_data = zlib.decompress(base64.b64decode(wav_data))
        rospy.loginfo("Calling")
        for data in self.stride(wav_data, int(2 * 16000 * self.stride_size / 1000), 2 * 16000):
            data = np.frombuffer(data, dtype=np.int16) / 32768.
            model_in = torch.from_numpy(self.compute_mfccs(data).squeeze(2)).unsqueeze(0)
            model_in = torch.autograd.Variable(model_in, requires_grad=False)
            model_in = model_in.cuda()
            predictions = F.softmax(self.model(model_in).squeeze(0).cpu()).data.numpy()
            pred_idx = np.argmax(predictions)
            pred_lbl = self.labels[pred_idx]
            pred_conf = np.max(predictions)

            if self.audio_lock.acquire(True):
                msg = kwsData()
                msg.cls = np.argmax(predictions)
                msg.conf = pred_conf
                msg.message = pred_lbl
                # msg.header.stamp = rospy.Time.now()
                self.out = msg
                self.audio_lock.release()


    def compute_mfccs(self, data):
        sr = 16000
        n_dct_filters = 40
        n_mels = 40
        f_max = 4000
        f_min = 20
        n_fft = 480
        hop_ms = 10
        hop_length = sr // 1000 * hop_ms
        dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        data = librosa.feature.melspectrogram(
            data,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            fmin=f_min,
            fmax=f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

    def stride(self, array, stride_size, window_size):
        i = 0
        while i + window_size <= len(array):
            yield array[i:i + window_size]
            i += stride_size

    def get_instances(self, event):
        if self.out is None:
            return
        self.audio_pub.publish(self.out)


if __name__ == '__main__':
    try:
        kws_model_path = "/home/lang/Ascent/project/whyvoice/src/kws/model/whyvoice_enhanced.pt"
        KWSNode(kws_model_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
