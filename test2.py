import cv2
import dlib
import os
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import paho.mqtt.client as mqtt
import io
import json
import string
import random
from PIL import Image
from threading import Thread
import queue
import time
from subprocess import Popen, PIPE


class RTSP_in:
    def __init__(self, source):
        self.in_queue = queue.Queue()
        self.source = source

        self.cap = cv2.VideoCapture(self.source)
        assert self.cap.isOpened(), 'Failed to open %s' % self.source

        _,_ = self.cap.read()

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) % 100

        self.input_thread = Thread(target=self.update, daemon=True)
        self.input_thread.start()
        print('Successfully opened %s (%gx%g at %.2f FPS).' % (self.source, self.w, self.h, self.fps))


    def update(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.in_queue.empty():
                try:
                    self.in_queue.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.in_queue.put(frame)

    def get_img(self):
        return self.in_queue.get()


class RTSP_out:
    def __init__(self, w, h, fps, output):
        # Output stream is images piped to ffmpeg
        self.output = output
        self.out_queue = queue.Queue()
        self.latest_frame = None
        self.proc = Popen([
            'ffmpeg',
            '-f', 'mjpeg',
            '-i', '-',
            '-c:v', 'copy',
            '-s', f'{w}x{h}',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.output], stdin=PIPE)
        self.output_thread = Thread(target=self.update, daemon=True)
        self.output_thread.start()
        print(f"\n *** Launched RTSP Streaming at {self.output} ***\n\n")

    def update(self):
        while True:
            imu = self.out_queue.get()
            im0i = Image.fromarray(imu, 'RGB')
            im0i.save(self.proc.stdin, 'JPEG')


    def send_img_out(self, im0):
        if not self.out_queue.empty():
            try:
                self.out_queue.get_nowait()   # discard previous (unprocessed) frame
            except queue.Empty:
                pass
        self.out_queue.put(im0)


class MQTT_client:
    def __init__(self, source, mqtt_address, mqtt_topic, emotion):
        self.mqtt_queue = queue.Queue() 
        self.source = source
        self.emotion = emotion
        self.mqtt_topic = mqtt_topic

        if ':' in mqtt_address:
            host, port = mqtt_address.split(':')
            port = int(port)
        else:
            host, port = mqtt_address, 1883
        self.mqtt_client_id = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(10)])
        self.mqtt_client = mqtt.Client(f"emotion_detection_{self.mqtt_client_id}")
        self.mqtt_client.connect(host, port=port)
        self.mqtt_last_message = [0, 0, 0, 0, 0, 0, 0]
        self.mqtt_last_message_time = time.time()
        self.mqtt_thread = Thread(target=self.update, daemon=True)
        self.mqtt_thread.start()

    def update(self):
        mqtt_new_message = self.mqtt_queue.get()
        # send update every 30s if nothing's happened
        if mqtt_new_message != self.mqtt_last_message or time.time() - self.mqtt_last_message_time > 30:
            mqtt_last_message = mqtt_new_message
            mqtt_last_message_time = time.time()

            # Build JSON message
            mqtt_json = {}
            mqtt_json['classes'] = []

            for i in range(len(self.emotion)):
                mqtt_json['classes'].append({'emotion_name': self.emotion[i], 'percentage': mqtt_new_message[i]})

            mqtt_json["input_rtsp_url"] = self.source

            # Send MQTT
            self.mqtt_client.publish(self.mqtt_topic, json.dumps(mqtt_json))
    
    def send_mqtt_msg(self, mqtt_new_message):
        self.mqtt_queue.put(mqtt_new_message)


def detect(source, output, mqtt_address, mqtt_topic):
    emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    print()
    print("INPUT RTSP URL\t-\t", source if len(source) > 0 else "None")
    print("OUTPUT RTSP URL\t-\t", output if len(output) > 0 else "None")
    print("MQTT URL\t-\t", mqtt_address if len(mqtt_address) > 0 else "None")
    print("MQTT TOPIC\t-\t", mqtt_topic if len(mqtt_topic) > 0 else "None")
    print()

    input_rtsp = RTSP_in(source)

    # initialize dlib's face detector (HOG-based) and then create the
    # facial landmark predictor
    print("[INFO] loading facial landmark predictor...")

    # Load detector
    face_detect = dlib.get_frontal_face_detector()

    # Load predictor
    # OR landmark_predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmark_predict = dlib.shape_predictor("Models/face_landmarks.dat")

    # Load emotion predictor
    model = load_model("Models/video.h5")

    if len(mqtt_address) > 0:
        mqtt_client = MQTT_client(source, mqtt_address, mqtt_topic, emotion)
    
    if len(output) > 0:
        output_rtsp = RTSP_out(input_rtsp.w, input_rtsp.h, input_rtsp.fps, output)

    while True:
        frame = input_rtsp.get_img()

        mqtt_new_message = [0, 0, 0, 0, 0, 0, 0]

        # Convert image into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use face_detect to find landmarks
        rects = face_detect(gray, 1)

        for (i, rect) in enumerate(rects):

            # compute the bounding box of the face and draw it on the frame
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y: y + h, x: x + w]

            # cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 100), 2)

            # Creates landmark object, then convert the facial landmark (x, y)-coordinates to a NumPy array
            landmarks = landmark_predict(image=gray, box=rect)
            landmarks = face_utils.shape_to_np(landmarks)

            face = zoom(face, (48 / face.shape[0], 48 / face.shape[1]))
            face = face.astype(np.float32)

            # Scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))

            # Make Prediction
            prediction = model.predict(face)
            prediction_result = np.argmax(prediction)

            for (i, (x, y)) in enumerate(landmarks):
                # Numbers
                cv2.putText(
                    frame,
                    str(i + 1),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                )
 
                for (j, k) in landmarks:
                    cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "Angry : " + str(round(prediction[0][0], 3)),
                (40, 180 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                0,
            )
            cv2.putText(
                frame,
                "Disgust : " + str(round(prediction[0][1], 3)),
                (40, 200 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                0,
            )
            cv2.putText(
                frame,
                "Fear : " + str(round(prediction[0][2], 3)),
                (40, 220 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Happy : " + str(round(prediction[0][3], 3)),
                (40, 240 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Sad : " + str(round(prediction[0][4], 3)),
                (40, 260 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Surprise : " + str(round(prediction[0][5], 3)),
                (40, 280 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )
            cv2.putText(
                frame,
                "Neutral : " + str(round(prediction[0][6], 3)),
                (40, 300 + 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                155,
                1,
            )

            # 2. Annotate main image with a label
            if prediction_result == 0:
                cv2.putText(
                    frame,
                    "Angry",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            elif prediction_result == 1:
                cv2.putText(
                    frame,
                    "Disgust",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            elif prediction_result == 2:
                cv2.putText(
                    frame,
                    "Fear",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            elif prediction_result == 3:
                cv2.putText(
                    frame,
                    "Happy",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            elif prediction_result == 4:
                cv2.putText(
                    frame,
                    "Sad",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            elif prediction_result == 5:
                cv2.putText(
                    frame,
                    "Surprise",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Neutral",
                    (x + w - 200, y - 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            mqtt_new_message = [
                str(round(prediction[0][0], 3)),
                str(round(prediction[0][1], 3)),
                str(round(prediction[0][2], 3)),
                str(round(prediction[0][3], 3)),
                str(round(prediction[0][4], 3)),
                str(round(prediction[0][5], 3)),
                str(round(prediction[0][6], 3))
            ]

        cv2.putText(
            frame,
            "Number of Faces : " + str(len(rects)),
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            155,
            1,
        )

        # Send new frame to output stream
        if len(output) > 0:
            output_rtsp.send_img_out(frame[:,:,::-1])

        # Publish detections to MQTT
        if len(mqtt_address) > 0:
            mqtt_client.send_mqtt_msg(mqtt_new_message)


if __name__ == '__main__':
    source = os.environ['INPUT_RTSP_URL']
    output = os.environ['OUTPUT_RTSP_URL']
    mqtt_address = os.environ['MQTT_URL']
    mqtt_topic = os.environ['MQTT_TOPIC']

    if len(source) == 0:
        raise Exception("Input rtsp stream must be specified ($INPUT_RTSP_URL)")
    if len(mqtt_address) > 0 and len(mqtt_topic) == 0:
        raise Exception("MQTT topic must be specified if address is given ($MQTT_TOPIC)")
    if len(mqtt_topic) > 0 and len(mqtt_address) == 0:
        raise Exception(
            "MQTT broker address must be specified if topic is specified ($MQTT_URL). Format = host or host:port (default port is 1883)")
    if not (source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))):
        raise Exception("The specified input is not a video stream.")
    if source.isnumeric():
        source = int(source)

    detect(source, output, mqtt_address, mqtt_topic)
